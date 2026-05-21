import time
import urllib.request

import ee
from pyproj import Transformer

# Autenticação e inicialização do Earth Engine
ee.Authenticate()
ee.Initialize(project="sentinel2-demo")

# Bounding box da região de Sinop-MT
bbox_sinop = [-55.62, -11.95, -55.40, -11.78]

# Transformar bounding box para EPSG:32721
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
minx, miny = transformer.transform(bbox_sinop[0], bbox_sinop[1])
maxx, maxy = transformer.transform(bbox_sinop[2], bbox_sinop[3])
roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:32721", geodesic=False)

# Buscar imagem Sentinel-2 (mediana)
image = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate("2025-06-01", "2025-08-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .median()
    .clip(roi)
)

# Input da composição colorida
band_input = input("Insira a composição de bandas (ex: 321, 432): ")
assert len(band_input) == 3, "Valor inválido, insira um valor com 3 dígitos"
band_indices = [int(b) for b in band_input]

t0 = time.time()

# Mapeamento número da banda -> mean band
BAND_MAP = {1: "B2", 2: "B3", 3: "B4", 4: "B8"}

# Selecionar bandas na ordem do input
selected_bands = [BAND_MAP[b] for b in band_indices]
img = image.select(selected_bands)

# Aplicar escala linear e correção gamma
BAND_MAX = {"B2": 3000, "B3": 3000, "B4": 3000, "B8": 5000}
maxs = ee.Image.constant([BAND_MAX[b] for b in selected_bands])
result = img.divide(maxs).clamp(0, 1).pow(0.75).multiply(255).toUint8()

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
