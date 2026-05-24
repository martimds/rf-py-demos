import urllib.request

import ee
import geopandas as gpd
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
    .select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
)

gdf = gpd.read_file("samplesentinel2025.gpkg").to_crs("EPSG:32721")
features = []
for _, row in gdf.iterrows():
    geom = ee.Geometry(
        row.geometry.__geo_interface__, proj="EPSG:32721", geodesic=False
    )
    features.append(ee.Feature(geom, {"uso_solo": int(row["uso_solo"])}))  # type: ignore
samples_fc = ee.FeatureCollection(features)

training = image.sampleRegions(collection=samples_fc, properties=["uso_solo"], scale=10)

classifier = ee.Classifier.smileGradientTreeBoost(numberOfTrees=100).train(
    features=training,
    classProperty="uso_solo",
    inputProperties=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
)

classified = image.classify(classifier).toUint8()

url = classified.getDownloadURL(
    {"region": roi, "scale": 10, "format": "GEO_TIFF", "crs": "EPSG:32721"}
)
output_name = "imagens/classif_sentinel2_2025_rf.tiff"
urllib.request.urlretrieve(url, output_name)
print(f"Salvo: {output_name} (10m, EPSG:32721)")
