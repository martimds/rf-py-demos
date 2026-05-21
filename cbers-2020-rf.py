import os

os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"

import geopandas as gpd
import numpy as np
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Criar serviço a partir da URL do STAC-INPE
service = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")

# Bounding box da região de Sinop-MT
bbox_sinop = [-55.62, -11.95, -55.40, -11.78]

# Buscar item CBERS4A WPM 2020
item_search = service.search(
    collections=["CB4A-WPM-L4-DN-1"],
    bbox=bbox_sinop,
    datetime="2020-01-01/2020-12-31",
    query={"eo:cloud_cover": {"lt": 10}},
)
item = list(item_search.items())[4]
print(f"Item: {item.id}")

# Converter bbox para EPSG:32721
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
minx, miny = transformer.transform(bbox_sinop[0], bbox_sinop[1])
maxx, maxy = transformer.transform(bbox_sinop[2], bbox_sinop[3])

# Resolução nativa das bandas multi (8m)
RES = 8.0
width = int((maxx - minx) // RES)
height = int((maxy - miny) // RES)
maxx = minx + width * RES
maxy = miny + height * RES
img_transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)

# Ler as 4 bandas originais (sem fusão, sem stretch)
BAND_NAMES = ["BAND1", "BAND2", "BAND3", "BAND4"]  # Blue, Green, Red, NIR
BAND_LABELS = ["Blue", "Green", "Red", "NIR"]

print("Lendo bandas multiespectrais (8m)...")
stack = np.zeros((4, height, width), dtype=np.float64)
for i, band_name in enumerate(BAND_NAMES):
    with rasterio.open(item.assets[band_name].href) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform).round_offsets().round_lengths()
        stack[i] = src.read(1, window=window, out_shape=(height, width))

# Calcular índices espectrais (features adicionais para melhorar separabilidade)
red = stack[2]
nir = stack[3]
green = stack[1]

# NDVI - separa vegetação de solo/água
ndvi = (nir - red) / (nir + red + 1e-10)

# NDWI - separa água de outros alvos
ndwi = (green - nir) / (green + nir + 1e-10)

# Stack final: 4 bandas + 2 índices = 6 features
stack_full = np.vstack([stack, ndvi[np.newaxis], ndwi[np.newaxis]])
FEATURE_LABELS = BAND_LABELS + ["NDVI", "NDWI"]

print(f"Features: {FEATURE_LABELS}")

# Carregar polígonos de treino (.gpkg com atributo "tipo")
gpkg_path = input("Caminho do .gpkg com polígonos de treino: ")
gdf = gpd.read_file(gpkg_path)

# Reprojetar para EPSG:32721 se necessário
if gdf.crs.to_epsg() != 32721:
    gdf = gdf.to_crs("EPSG:32721")

# Mapear classes para inteiros
classes = sorted(gdf["tipo"].unique())
class_map = {name: idx + 1 for idx, name in enumerate(classes)}
print(f"Classes: {class_map}")

# Rasterizar polígonos
shapes = [(geom, class_map[tipo]) for geom, tipo in zip(gdf.geometry, gdf["tipo"])]
labels = rasterize(shapes, out_shape=(height, width), transform=img_transform, fill=0, dtype=np.uint8)

# Extrair pixels de treino (onde labels > 0)
mask = labels > 0
X = stack_full[:, mask].T  # (n_pixels, 6 features)
y = labels[mask]

print(f"Pixels de treino: {X.shape[0]} ({X.shape[1]} features)")
for cls_name, cls_id in class_map.items():
    print(f"  {cls_name}: {(y == cls_id).sum()} pixels")

# Validação cruzada estratificada (5-fold)
print("\nValidação cruzada (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=5,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"Acurácia CV: {scores.mean():.4f} ± {scores.std():.4f}")

# Treinar modelo final com todos os dados
rf.fit(X, y)
print(f"Acurácia OOB: {rf.oob_score_:.4f}")

# Importância das features
print("\nImportância das features:")
for label, imp in sorted(zip(FEATURE_LABELS, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {label}: {imp:.4f}")

# Relatório de classificação (OOB predictions)
y_oob = np.argmax(rf.oob_decision_function_, axis=1) + 1
print(f"\nRelatório de classificação (OOB):")
print(classification_report(y, y_oob, target_names=classes))

# Matriz de confusão
print("Matriz de confusão (OOB):")
cm = confusion_matrix(y, y_oob)
print(f"{'':>12}", end="")
for c in classes:
    print(f"{c:>12}", end="")
print()
for i, row in enumerate(cm):
    print(f"{classes[i]:>12}", end="")
    for val in row:
        print(f"{val:>12}", end="")
    print()

# Classificar imagem inteira
print("\nClassificando imagem...")
X_all = stack_full.reshape(len(FEATURE_LABELS), -1).T
y_pred = rf.predict(X_all).reshape(height, width).astype(np.uint8)

# Salvar mapa classificado
output_name = os.path.join("imagens", "cbers2020_classificacao_rf.tiff")
profile = {
    "driver": "GTiff",
    "dtype": "uint8",
    "width": width,
    "height": height,
    "count": 1,
    "crs": "EPSG:32721",
    "transform": img_transform,
}

with rasterio.open(output_name, "w", **profile) as dst:
    dst.write(y_pred, 1)

print(f"\nSalvo: {output_name} ({width}x{height} px, 8m)")
print(f"Legenda: {class_map}")
