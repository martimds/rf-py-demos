import concurrent.futures
import os
import time

os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

import geopandas as gpd
import numpy as np
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds
from sklearn.ensemble import RandomForestClassifier

bbox_sinop = [-55.62, -11.95, -55.40, -11.78]

service = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")
item_search = service.search(
    collections=["CB4A-MUX-L4-SR-1"],
    bbox=bbox_sinop,
    datetime="",
    query={"eo:cloud_cover": {"lt": 10}},
)
item = list(item_search.items())[0]

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
minx, miny = transformer.transform(bbox_sinop[0], bbox_sinop[1])
maxx, maxy = transformer.transform(bbox_sinop[2], bbox_sinop[3])

TARGET_RES = 20.0
width = int((maxx - minx) // TARGET_RES)
height = int((maxy - miny) // TARGET_RES)
maxx = minx + width * TARGET_RES
maxy = miny + height * TARGET_RES
target_transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)

BAND_MAP = {0: "BAND5", 1: "BAND6", 2: "BAND7", 3: "BAND8"}

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def download_crop(key):
    local_path = os.path.join(CACHE_DIR, f"{item.id}_{key}_crop.tif")

    if os.path.exists(local_path):
        print(f"Cache hit: {key}")
        return local_path

    print(f"Baixando recorte de {key}...")
    with rasterio.open(item.assets[key].href) as src:
        window = (
            from_bounds(minx, miny, maxx, maxy, src.transform)
            .round_offsets()
            .round_lengths()
        )
        data = src.read(
            1,
            window=window,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            transform=target_transform,
        )

        with rasterio.open(local_path, "w", **profile) as dst:
            dst.write(data, 1)

    print(f"Concluído: {key}")
    return local_path


def read_band(local_path):
    with rasterio.open(local_path) as src:
        return src.read(1).astype(np.float32)


t0 = time.time()

print("Lendo bandas CBERS-4A MUX 2025...")
assets = [BAND_MAP[b] for b in range(4)]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(download_crop, key) for key in assets]
    local_paths = [f.result() for f in futures]

print(f"Download concluído em {time.time() - t0:.1f}s. Processando...")

bands = np.stack([read_band(p) for p in local_paths])  # (4, H, W)

gdf = gpd.read_file("samplecbers2025.gpkg").to_crs("EPSG:32721")

labels = rasterize(
    [(geom, val) for geom, val in zip(gdf.geometry, gdf["uso_solo"])],
    out_shape=(height, width),
    transform=target_transform,
    fill=0,
    dtype=np.uint8,
)

mask = labels > 0  # type: ignore
X_train = bands[:, mask].T
y_train = labels[mask]  # type: ignore

print(f"Treinando RF com {X_train.shape[0]} pixels...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

X_all = bands.reshape(4, -1).T
y_pred = clf.predict(X_all).reshape(height, width).astype(np.uint8)

os.makedirs("imagens", exist_ok=True)
output_name = os.path.join("imagens", "classif_cbers4a_2025_rf.tiff")
profile = {
    "driver": "GTiff",
    "dtype": "uint8",
    "width": width,
    "height": height,
    "count": 1,
    "crs": "EPSG:32721",
    "transform": target_transform,
}
with rasterio.open(output_name, "w", **profile) as dst:
    dst.write(y_pred, 1)

print(f"Salvo: {output_name} ({width}x{height} px, 20m)")
print(f"Tempo total: {time.time() - t0:.1f}s")
