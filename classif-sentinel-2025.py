import urllib.request

import ee
import geopandas as gpd
from pyproj import Transformer

# Autenticação e inicialização do Earth Engine
ee.Authenticate()
ee.Initialize(project="sentinel2-demo")

# Bounding box da região de Sinop-MT
bbox_sinop = [-55.62, -11.95, -55.40, -11.78]

# Converter bounding box para EPSG:32721
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
minx, miny = transformer.transform(bbox_sinop[0], bbox_sinop[1])
maxx, maxy = transformer.transform(bbox_sinop[2], bbox_sinop[3])
roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:32721", geodesic=False)

# Imagem Sentinel-2
image = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate("2025-06-01", "2025-08-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .median()
    .clip(roi)
    .select(["B2", "B3", "B4", "B8"])
)

# Carregar GeoPackage e converter para FeatureCollection
gdf = gpd.read_file("samplesentinel2025.gpkg").to_crs("EPSG:32721")
features = []
for _, row in gdf.iterrows():
    geom = ee.Geometry(
        row.geometry.__geo_interface__, proj="EPSG:32721", geodesic=False
    )
    features.append(ee.Feature(geom, {"uso_solo": int(row["uso_solo"])}))  # type: ignore
samples_fc = ee.FeatureCollection(features)

# Extrair valores espectrais das amostras
training = image.sampleRegions(collection=samples_fc, properties=["uso_solo"], scale=10)

# Treinar classificador smileRandomForest
classifier = ee.Classifier.smileRandomForest(numberOfTrees=100).train(
    features=training,
    classProperty="uso_solo",
    inputProperties=["B2", "B3", "B4", "B8"],
)

# Classificar imagem
classified = image.classify(classifier).toUint8()

# Baixar resultado
url = classified.getDownloadURL(
    {"region": roi, "scale": 10, "format": "GEO_TIFF", "crs": "EPSG:32721"}
)
output_name = "imagens/classif_sentinel2_2025_rf.tiff"
urllib.request.urlretrieve(url, output_name)
print(f"Salvo: {output_name} (10m, EPSG:32721)")
