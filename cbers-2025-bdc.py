import concurrent.futures
import os
import time

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.tiff"

import numpy as np
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds

# Criar serviço a partir da URL do STAC-INPE
service = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")

# Bounding box da região de Sinop-MT
bbox_sinop = [-55.62, -11.95, -55.40, -11.78]

# Criar busca no PyStac
item_search = service.search(
    collections=["CB4A-WPM-L4-DN-1"],
    bbox=bbox_sinop,
    datetime="2025-06-01/2025-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
)

# Selecionar o item desejado
item = list(item_search.items())[0]

# Converter bounding box para CRS da imagem (EPSG:32721)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
minx, miny = transformer.transform(bbox_sinop[0], bbox_sinop[1])
maxx, maxy = transformer.transform(bbox_sinop[2], bbox_sinop[3])

# Resolução alvo
TARGET_RES = 2.0

# Calcular tamanho do output
width = int((maxx - minx) // TARGET_RES)
height = int((maxy - miny) // TARGET_RES)
maxx = minx + width * TARGET_RES
maxy = miny + height * TARGET_RES
target_transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)

# Input da composição
band_input = input("Insira a composição de bandas (ex: 321, 432): ")
assert len(band_input) == 3, "Valor inválido, insira um valor com 3 dígitos: "
band_indices = [int(b) for b in band_input]

t0 = time.time()

# Mapeamento número da banda -> nome do asset
BAND_MAP = {1: "BAND1", 2: "BAND2", 3: "BAND3", 4: "BAND4"}

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Criar lista de assets e reamostragens
assets = [BAND_MAP[b] for b in band_indices] + ["BAND0"]
resamplings = [Resampling.lanczos] * 3 + [Resampling.nearest]


# Baixar recorte remoto e salva no diretorio cache
def download_crop(key, resampling):
    local_path = os.path.join(CACHE_DIR, f"{item.id}_{key}_crop.tif")

    # Verificar se o arquivo ja existe no cache
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
            resampling=resampling,
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


# Baixar recortes em paralelo
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(download_crop, key, r) for key, r in zip(assets, resamplings)
    ]
    local_paths = [f.result() for f in futures]

print("Download concluído. Processando...")


# Ler banda do cache local
def read_band(local_path):
    with rasterio.open(local_path) as src:
        return src.read(1).astype(np.float32)


bands = [read_band(p) for p in local_paths[:3]]
pan = read_band(local_paths[3])

# Fusão IHS (FastIHS)
intensity = (bands[0] + bands[1] + bands[2]) / 3.0
pan_matched = pan * (intensity.mean() / pan.mean())
delta = pan_matched - intensity


# % Stretch e conversão para 8 bits
def normalize(arr):
    low, high = np.percentile(arr, (2, 98))
    normalized = (arr - low) / (high - low)
    return (np.clip(normalized, 0, 1) * 255).astype(np.uint8)


# Aplicar delta e normalizar
fused = [normalize(b + delta) for b in bands]

# Salvar TIFF
os.makedirs("imagens", exist_ok=True)
output_name = os.path.join("imagens", f"cbers4a_2025_{band_input}.tiff")
profile = {
    "driver": "GTiff",
    "dtype": "uint8",
    "width": width,
    "height": height,
    "count": 3,
    "crs": "EPSG:32721",
    "transform": target_transform,
}

with rasterio.open(output_name, "w", **profile) as dst:
    for i, band in enumerate(fused, 1):
        dst.write(band, i)
    dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

print(f"Salvo: {output_name} ({width}x{height} px, {TARGET_RES}m, EPSG:32721)")
print(f"Tempo corrido: {time.time() - t0:.1f}s")
