import ee

ee.Authenticate()
ee.Initialize(project="sentinel2-demo")

# Região de interesse (Sinop-MT)
roi = ee.Geometry.Rectangle([-55.62, -11.95, -55.40, -11.78])

# Buscar imagem Sentinel-2 com menor cobertura de nuvens
image = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate("2020-01-01", "2020-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .sort("CLOUDY_PIXEL_PERCENTAGE")
    .first()
    .clip(roi)
)
