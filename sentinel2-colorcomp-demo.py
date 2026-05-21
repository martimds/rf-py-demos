import urllib.request

import ee

ee.Authenticate()
ee.Initialize(project="sentinel2-demo")

# Bounding box da região de Sinop-MT
bbox_sinop = [-55.62, -11.95, -55.40, -11.78]
roi = ee.Geometry.Rectangle(bbox_sinop)

# Criar busca de imagens do Sentinel-2
image = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate("2020-01-01", "2020-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .sort("CLOUDY_PIXEL_PERCENTAGE")
    .first()
    .clip(roi)
)

# Input da composição colorida
band_input = input("Insira a composição de bandas (ex: 321, 432): ")
assert len(band_input) == 3, "Valor inválido, insira um valor com 3 dígitos"
band_indices = [int(b) for b in band_input]

# Mapeamento número da banda -> nome Sentinel-2
BAND_MAP = {1: "B2", 2: "B3", 3: "B4", 4: "B8"}

# Selecionar bandas na ordem do input
selected_bands = [BAND_MAP[b] for b in band_indices]
img = image.select(selected_bands)

# Percentile stretch para 8 bits (processado no servidor)
percentiles = img.reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]),
    geometry=roi,
    scale=10,
    maxPixels=1e9,
).getInfo()

stretched_bands = []
for band_name in selected_bands:
    low = percentiles[f"{band_name}_p2"]
    high = percentiles[f"{band_name}_p98"]
    stretched = (
        img.select(band_name)
        .subtract(low)
        .divide(high - low)
        .clamp(0, 1)
        .multiply(255)
        .toUint8()
    )
    stretched_bands.append(stretched)

result = ee.Image.cat(stretched_bands)

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
