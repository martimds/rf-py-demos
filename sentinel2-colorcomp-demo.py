import time
import urllib.request

import ee
from pyproj import Transformer

ee.Authenticate()
ee.Initialize(project="sentinel2-demo")

# Bounding box da região de Sinop-MT (EPSG:4326 -> EPSG:32721)
bbox_sinop = [-55.62, -11.95, -55.40, -11.78]

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
minx, miny = transformer.transform(bbox_sinop[0], bbox_sinop[1])
maxx, maxy = transformer.transform(bbox_sinop[2], bbox_sinop[3])
roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:32721", geodesic=False)

# Buscar imagem Sentinel-2 (mediana para remover nuvens)
image = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate("2020-01-01", "2020-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .median()
    .clip(roi)
)

# Input da composição colorida
band_input = input("Insira a composição de bandas (ex: 321, 432): ")
assert len(band_input) == 3, "Valor inválido, insira um valor com 3 dígitos"
band_indices = [int(b) for b in band_input]

t0 = time.time()

# Mapeamento número da banda -> banda sentinel-2
BAND_MAP = {1: "B2", 2: "B3", 3: "B4", 4: "B8"}

# Selecionar bandas na ordem do input
selected_bands = [BAND_MAP[b] for b in band_indices]
img = image.select(selected_bands)

# Equalização via CDF no servidor (equivalente a histogram equalization)
# Usa percentis densos para construir uma lookup table de equalização
percentile_list = list(range(0, 101))
percentiles = img.reduceRegion(
    reducer=ee.Reducer.percentile(percentile_list),
    geometry=roi,
    scale=10,
    maxPixels=1e9,
).getInfo()

stretched_bands = []
for band_name in selected_bands:
    band = img.select(band_name)
    # Construir stretch cumulativo: mapear cada faixa de percentil para 0-255
    result_band = ee.Image(0).toUint8()
    for i in range(100):
        low = percentiles[f"{band_name}_p{i}"]
        high = percentiles[f"{band_name}_p{i + 1}"]
        if high > low:
            segment = (
                band.subtract(low)
                .divide(high - low)
                .clamp(0, 1)
                .multiply(255 * (i + 1) / 100 - 255 * i / 100)
                .add(255 * i / 100)
            )
            mask = band.gte(low).And(band.lt(high))
            result_band = result_band.where(mask, segment)
    # Último percentil
    result_band = result_band.where(band.gte(percentiles[f"{band_name}_p100"]), 255)
    stretched_bands.append(result_band.toUint8())

result = ee.Image.cat(stretched_bands).divide(255).pow(0.3).multiply(255).toUint8()

# Baixar e salvar TIFF
url = result.getDownloadURL(
    {
        "region": roi,
        "scale": 10,
        "format": "GEO_TIFF",
        "crs": "EPSG:32721",
    }
)

output_name = f"imagens/sentinel2_colorcomp_{band_input}.tiff"
urllib.request.urlretrieve(url, output_name)
print(f"Salvo: {output_name} (10m, EPSG:32721)")
print(f"Tempo corrido: {time.time() - t0:.1f}s")
