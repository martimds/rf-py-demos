import time
import urllib.request

import ee
from pyproj import Transformer

ee.Authenticate()
ee.Initialize(project="sentinel2-demo")

bbox_sinop = [-55.62, -11.95, -55.40, -11.78]

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
minx, miny = transformer.transform(bbox_sinop[0], bbox_sinop[1])
maxx, maxy = transformer.transform(bbox_sinop[2], bbox_sinop[3])
roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:32721", geodesic=False)

image = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate("2025-01-01", "2025-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .filter(ee.Filter.contains(".geo", roi))
    .sort("CLOUDY_PIXEL_PERCENTAGE")
    .first()
    .clip(roi)
)

band_input = input("Insira a composição de bandas (ex: 321, 432): ")
assert len(band_input) == 3, "Valor inválido, insira um valor com 3 dígitos"
band_indices = [int(b) for b in band_input]

t0 = time.time()

BAND_MAP = {1: "B2", 2: "B3", 3: "B4", 4: "B8"}
selected_bands = [BAND_MAP[b] for b in band_indices]
img = image.select(selected_bands)

stats = img.reduceRegion(
    reducer=ee.Reducer.percentile([2, 98]),
    geometry=image.geometry(),
    scale=10,
    maxPixels=1e13,
)

lows = ee.Image.constant([stats.getNumber(f"{b}_p2") for b in selected_bands])
highs = ee.Image.constant([stats.getNumber(f"{b}_p98") for b in selected_bands])

GAMMA_MAP = {"B2": 0.9, "B3": 0.6, "B4": 0.8, "B8": 0.8}
gammas = ee.Image.constant([GAMMA_MAP[b] for b in selected_bands])

result = (
    img.subtract(lows)
    .divide(highs.subtract(lows))
    .clamp(0, 1)
    .pow(gammas)
    .multiply(255)
    .toUint8()
)

url = result.getDownloadURL(
    {
        "region": roi,
        "scale": 10,
        "format": "GEO_TIFF",
        "crs": "EPSG:32721",
    }
)

output_name = f"imagens/sentinel2_2025_{band_input}.tiff"
urllib.request.urlretrieve(url, output_name)
print(f"Salvo: {output_name} (10m, EPSG:32721)")
print(f"Tempo corrido: {time.time() - t0:.1f}s")
