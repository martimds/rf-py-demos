import os
import shutil
import urllib.request

import ee
import numpy as np
import pystac_client
import rasterio
from arosics import COREG
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds

TARGET_CRS = CRS.from_epsg(31981)
TARGET_RES = 2.0

minx, miny = 654842.1, 8689388.3
maxx, maxy = 661578.1, 8693384.9

_transformer = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
bbox_sinop = [*_transformer.transform(minx, miny), *_transformer.transform(maxx, maxy)]

RAW_CBERS = os.path.join("RAW", "CBERS4A")
RAW_SENTINEL = os.path.join("RAW", "SENTINEL2")
CROP_DIR = os.path.join("RAW", "CBERS4A", "crop")
ALIGNED_DIR = "IMAGES"

for d in [RAW_CBERS, RAW_SENTINEL, CROP_DIR, ALIGNED_DIR]:
    os.makedirs(d, exist_ok=True)

print("STEP 1: Baixar CBERS4A pelo STAC BDC")

service = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")

CBERS_DATES = {"2020": "2020-07-21/2020-07-21", "2025": "2025-07-24/2025-07-24"}
CBERS_BANDS = ["BAND0", "BAND1", "BAND2", "BAND3", "BAND4"]


def download_cbers_band(item, key):
    url = item.assets[key].href
    local_path = os.path.join(RAW_CBERS, f"{item.id}_{key}.tif")

    if os.path.exists(local_path):
        return local_path

    print(f"Baixando {key}...")
    urllib.request.urlretrieve(url, local_path)

    return local_path


def download_all_cbers():
    items = {}
    for year, date_range in CBERS_DATES.items():
        search = service.search(
            collections=["CB4A-WPM-L4-DN-1"],
            bbox=bbox_sinop,
            datetime=date_range,
        )
        item = list(search.items())[0]
        items[year] = item
        print(f"{year}: {item.id} ({item.datetime})")

    for year, item in items.items():
        print(f"Baixando CBERS {year}...")
        for b in CBERS_BANDS:
            download_cbers_band(item, b)

    return items


cbers_items = download_all_cbers()
print("(CBERS) Download concluído.")

print("STEP 2: Baixar Sentinel-2 via GEE")

ee.Authenticate()
ee.Initialize(project="sentinel2-demo")

roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:31981", geodesic=False)

S2_BANDS = ["B2", "B3", "B4", "B8"]
S2_DATES = {"2020": ("2020-07-01", "2020-07-31"), "2025": ("2025-07-01", "2025-07-31")}

for year, (start, end) in S2_DATES.items():
    print(f"Sentinel-2 {year}...")
    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
        .median()
        .clip(roi)
    )

    for band in S2_BANDS:
        output_path = os.path.join(RAW_SENTINEL, f"sentinel2_{year}_july_{band}.tiff")
        if os.path.exists(output_path):
            print(f"Já existe: {band}")
            continue
        url = (
            image.select(band)
            .toUint16()
            .getDownloadURL(
                {"region": roi, "scale": 10, "format": "GEO_TIFF", "crs": "EPSG:31981"}
            )
        )
        print(f"Baixando {band}...")
        urllib.request.urlretrieve(url, output_path)

    stacked_path = os.path.join(RAW_SENTINEL, f"sentinel2_{year}_july_median.tiff")
    if not os.path.exists(stacked_path):
        with rasterio.open(
            os.path.join(RAW_SENTINEL, f"sentinel2_{year}_july_{S2_BANDS[0]}.tiff")
        ) as src:
            profile = src.profile.copy()
            profile.update(count=len(S2_BANDS))
        with rasterio.open(stacked_path, "w", **profile) as dst:
            for i, band in enumerate(S2_BANDS, 1):
                with rasterio.open(
                    os.path.join(RAW_SENTINEL, f"sentinel2_{year}_july_{band}.tiff")
                ) as src:
                    dst.write(src.read(1), i)
        print(f"Stack: {stacked_path}")

print("(Sentinel-2) Download concluído.")

print("STEP 3: Recortar CBERS")

for year, item in cbers_items.items():
    print(f"Recortando CBERS {year}...")
    for key in CBERS_BANDS:
        src_path = os.path.join(RAW_CBERS, f"{item.id}_{key}.tif")
        dst_path = os.path.join(CROP_DIR, f"{item.id}_{key}.tif")

        if os.path.exists(dst_path):
            continue

        with rasterio.open(src_path) as src:
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            data = src.read(1, window=window)
            profile = src.profile.copy()
            profile.update(
                height=data.shape[0],
                width=data.shape[1],
                transform=src.window_transform(window),
                compress=None,
            )
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data, 1)

        print(f"{key} recortado.")

print("Recorte concluído.")

print("STEP 4: Fusão CBERS (Gram-Schmidt)")


def pansharpen_gs(item_id, output_name):
    output_path = os.path.join(CROP_DIR, output_name)
    if os.path.exists(output_path):
        print(f"Já existe: {output_name}")
        return output_path

    pan_path = os.path.join(CROP_DIR, f"{item_id}_BAND0.tif")
    with rasterio.open(pan_path) as src:
        pan = src.read(1).astype(np.float32)
        pan_profile = src.profile.copy()
        pan_height, pan_width = pan.shape

    ms_bands = []
    for b in ["BAND1", "BAND2", "BAND3", "BAND4"]:
        with rasterio.open(os.path.join(CROP_DIR, f"{item_id}_{b}.tif")) as src:
            data = src.read(
                1, out_shape=(pan_height, pan_width), resampling=Resampling.lanczos
            )
            ms_bands.append(data.astype(np.float32))

    ms = np.array(ms_bands)
    n_bands = ms.shape[0]
    flat_ms = ms.reshape(n_bands, -1).T
    flat_pan = pan.ravel()

    weights = np.linalg.lstsq(flat_ms, flat_pan, rcond=None)[0]
    pan_sim = (ms * weights[:, None, None]).sum(axis=0)

    pan_adj = (pan - pan.mean()) * (pan_sim.std() / (pan.std() + 1e-8)) + pan_sim.mean()

    gs_components = [pan_sim]
    proj_coeffs = np.zeros((n_bands, n_bands + 1))
    for i in range(n_bands):
        band = ms[i].copy()
        for j, gs in enumerate(gs_components):
            c = np.sum(band * gs) / (np.sum(gs * gs) + 1e-8)
            proj_coeffs[i, j] = c
            band = band - c * gs
        gs_components.append(band)

    gs_components[0] = pan_adj

    result = np.zeros_like(ms)
    for i in range(n_bands):
        reconstructed = np.zeros_like(pan)
        for j in range(len(gs_components)):
            if j <= i:
                reconstructed += proj_coeffs[i, j] * gs_components[j]
            if j == i + 1:
                reconstructed += gs_components[j]
        result[i] = reconstructed

    result = np.clip(result, 0, None).astype(np.uint16)

    pan_profile.update(count=4, dtype="uint16", compress=None)
    with rasterio.open(output_path, "w", **pan_profile) as dst:
        for i in range(4):
            dst.write(result[i], i + 1)

    print(f"{output_name} ({pan_width}x{pan_height}, 2m)")
    return output_path


psp_paths = {}
for year, item in cbers_items.items():
    psp_paths[year] = pansharpen_gs(item.id, f"CBERS{year}_PSP.tif")

print("Fusão concluída.")

print("STEP 5: Reprojetar para EPSG:31981 (2m)")

ref_src_path = psp_paths["2020"]

with rasterio.open(ref_src_path) as src:
    ref_transform, ref_width, ref_height = calculate_default_transform(
        src.crs, TARGET_CRS, src.width, src.height, *src.bounds, resolution=TARGET_RES
    )

print(f"Grid de referência: {ref_width}x{ref_height}, 2m, {TARGET_CRS}")


def reproject_to_grid(src_path, dst_path):
    if os.path.exists(dst_path):
        print(f"Já existe: {os.path.basename(dst_path)}")
        return
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            width=ref_width,
            height=ref_height,
            transform=ref_transform,
            crs=TARGET_CRS,
            compress=None,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.lanczos,
                )
    print(f"Reprojetado: {os.path.basename(dst_path)}")


ref_path = os.path.join(ALIGNED_DIR, "cbers4a_2020_ref.tiff")
reproject_to_grid(psp_paths["2020"], ref_path)

reproj_files = {
    "cbers4a_2025": psp_paths["2025"],
    "sentinel2_2020": os.path.join(RAW_SENTINEL, "sentinel2_2020_july_median.tiff"),
    "sentinel2_2025": os.path.join(RAW_SENTINEL, "sentinel2_2025_july_median.tiff"),
}

reproj_paths = {}
for name, src_path in reproj_files.items():
    dst_path = os.path.join(ALIGNED_DIR, f"{name}_reproj.tiff")
    reproject_to_grid(src_path, dst_path)
    reproj_paths[name] = dst_path

print("Reprojeção concluída.")

print("STEP 6: (AROSICS) template-matching")


def arosics_align(target_path, out_path):
    if os.path.exists(out_path):
        print(f"Já existe: {os.path.basename(out_path)}")
        return

    CR = COREG(
        ref_path,
        target_path,
        path_out=out_path,
        fmt_out="GTIFF",
        ws=(1024, 1024),
        max_shift=20,
        max_iter=15,
        nodata=(0, 0),
        align_grids=True,
    )
    try:
        CR.calculate_spatial_shifts()
    except Exception as e:
        raise SystemExit(f"Erro ao alinhar {os.path.basename(target_path)}: {e}")

    if CR.ssim_improved:
        CR.correct_shifts()
        print(
            f"{os.path.basename(out_path)}: x={CR.x_shift_px:.2f}, y={CR.y_shift_px:.2f} px"
        )
    else:
        shutil.copy(target_path, out_path)
        print(f"{os.path.basename(out_path)}: já alinhado.")


aligned_paths = {"cbers4a_2020": ref_path}

for name, reproj_path in reproj_paths.items():
    out_path = os.path.join(ALIGNED_DIR, f"{name}_aligned.tiff")
    print(f"Alinhando {name}...")
    arosics_align(reproj_path, out_path)
    aligned_paths[name] = out_path

print("Alinhamento concluído.")

print("STEP 7: NDVI + NDWI")

for name, path in aligned_paths.items():
    out_path = os.path.join(ALIGNED_DIR, f"{name}_indices.tiff")
    if os.path.exists(out_path):
        print(f"Já existe: {name}_indices.tiff")
        continue

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()

    red, nir, green = data[2], data[3], data[1]
    ndvi = np.clip((nir - red) / (nir + red + 1e-8), -1, 1)
    ndwi = np.clip((green - nir) / (green + nir + 1e-8), -1, 1)

    profile.update(count=6, dtype="float32", compress="zstd")
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(4):
            dst.write(data[i], i + 1)
        dst.write(ndvi, 5)
        dst.write(ndwi, 6)

    print(f"{name}_indices.tiff (6 bandas: B,G,R,NIR,NDVI,NDWI)")

for f in os.listdir(ALIGNED_DIR):
    if not f.endswith("_indices.tiff"):
        os.remove(os.path.join(ALIGNED_DIR, f))

print("PIPELINE CONCLUÍDO")
print(f"Resultados em {ALIGNED_DIR}/")
