import os

os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"

import time

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
    datetime="2020-01-01/2020-12-31",
    query={"eo:cloud_cover": {"lt": 10}},
)

# Selecionar o item desejado
item = list(item_search.items())[4]

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


# Retornar banda multi recortada e reamostrada (lanczos)
def read_band_resampled(asset_key, target_height, target_width):
    with rasterio.open(item.assets[asset_key].href) as src:
        window = (
            from_bounds(minx, miny, maxx, maxy, src.transform)
            .round_offsets()
            .round_lengths()
        )
        data = src.read(
            1,
            window=window,
            out_shape=(target_height, target_width),
            resampling=Resampling.lanczos,
        )
    return data.astype(np.float64)


print(f"Lendo bandas {band_indices} e PAN, reamostrando para 4m...")

# Ler as 3 bandas selecionadas
bands = [read_band_resampled(BAND_MAP[b], height, width) for b in band_indices]

# Ler banda pan (vizinho mais proximo)
with rasterio.open(item.assets["BAND0"].href) as src:
    window = (
        from_bounds(minx, miny, maxx, maxy, src.transform)
        .round_offsets()
        .round_lengths()
    )
    pan = src.read(
        1, window=window, out_shape=(height, width), resampling=Resampling.nearest
    ).astype(np.float64)

# Fusão IHS (FastIHS)
intensity = (bands[0] + bands[1] + bands[2]) / 3.0
pan_matched = pan * (intensity.mean() / pan.mean())
delta = pan_matched - intensity


# Percentile stretch para 8 bits
def normalize(arr):
    low, high = np.percentile(arr, (2, 98))
    normalized = (arr - low) / (high - low)
    return (np.clip(normalized, 0, 1) * 255).astype(np.uint8)


# Normalizar bandas
fused = [normalize(b + delta) for b in bands]

# Salvar TIFF (preserva metadados e georreferencial)
output_name = os.path.join("imagens", f"cbers4a_fused_ihs_wpm_{band_input}.tiff")
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

print(f"Salvo: {output_name} ({width}x{height} px, 4m, EPSG:32721)")
print(f"Tempo corrido: {time.time() - t0:.1f}s")
