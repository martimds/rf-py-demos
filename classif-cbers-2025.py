import os

os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"

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

# Bounding box Sinop-MT
bbox_sinop = [-55.62, -11.95, -55.40, -11.78]

# Buscar imagem CBERS-4A 2025
service = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")
item_search = service.search(
    collections=["CB4A-WPM-L4-DN-1"],
    bbox=bbox_sinop,
    datetime="2025-06-01/2025-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
)
item = list(item_search.items())[0]

# Converter bbox para EPSG:32721
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
minx, miny = transformer.transform(bbox_sinop[0], bbox_sinop[1])
maxx, maxy = transformer.transform(bbox_sinop[2], bbox_sinop[3])

# Grid 8m (resolução nativa multiespectrais)
TARGET_RES = 8.0
width = int((maxx - minx) // TARGET_RES)
height = int((maxy - miny) // TARGET_RES)
maxx = minx + width * TARGET_RES
maxy = miny + height * TARGET_RES
target_transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)

# Ler 4 bandas multiespectrais
BAND_MAP = {1: "BAND1", 2: "BAND2", 3: "BAND3", 4: "BAND4"}


def read_band(asset_key):
    with rasterio.open(item.assets[asset_key].href) as src:
        window = (
            from_bounds(minx, miny, maxx, maxy, src.transform)
            .round_offsets()
            .round_lengths()
        )
        return src.read(
            1, window=window, out_shape=(height, width), resampling=Resampling.nearest
        ).astype(np.float32)


print("Lendo bandas CBERS-4A 2025...")
bands = np.stack([read_band(BAND_MAP[b]) for b in range(1, 5)])  # (4, H, W)

# Carregar amostras
gdf = gpd.read_file("samplecbers2025.gpkg").to_crs("EPSG:32721")

# Rasterizar amostras
labels = rasterize(
    [(geom, val) for geom, val in zip(gdf.geometry, gdf["uso_solo"])],
    out_shape=(height, width),
    transform=target_transform,
    fill=0,
    dtype=np.uint8,
)

# Extrair pixels de treino
mask = labels > 0  # type: ignore
X_train = bands[:, mask].T
y_train = labels[mask]  # type: ignore

# Treinar Random Forest
print(f"Treinando RF com {X_train.shape[0]} pixels...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Classificar imagem inteira
X_all = bands.reshape(4, -1).T
y_pred = clf.predict(X_all).reshape(height, width).astype(np.uint8)

# Salvar resultado
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

print(f"Salvo: {output_name} ({width}x{height} px, 8m)")
