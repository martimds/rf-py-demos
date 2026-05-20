import os

# Habilita os caches do GDAL (Opcional)
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from sklearn.ensemble import RandomForestClassifier

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

itens = list(item_search.items())

# Exibir itens encontrados
for idx, item in enumerate(itens):
    print(f"Index [{idx}] -> ID da Imagem: {item.id}")

# Selecionar o item desejado
item = itens[4]
print(f"Assets: {list(item.assets.keys())}")


# Converter bbox para o CRS da imagem (4326 -> 32721) para recorte
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32721", always_xy=True)
minx, miny = transformer.transform(bbox_sinop[0], bbox_sinop[1])
maxx, maxy = transformer.transform(bbox_sinop[2], bbox_sinop[3])


# Retorna recorte da imagem
def get_window(src):
    return (
        from_bounds(minx, miny, maxx, maxy, src.transform)
        .round_offsets()
        .round_lengths()
    )


# Obtem as imagens cortadas para cada banda (B, G, R, NIR)
with rasterio.open(item.assets["BAND1"].href) as src:
    window = get_window(src)
    blue = src.read(1, window=window).astype(float)
    win_transform = src.window_transform(window)

with rasterio.open(item.assets["BAND2"].href) as src:
    green = src.read(1, window=window).astype(float)

with rasterio.open(item.assets["BAND3"].href) as src:
    red = src.read(1, window=window).astype(float)

with rasterio.open(item.assets["BAND4"].href) as src:
    nir = src.read(1, window=window).astype(float)

# Stack de bandas (linhas, colunas, 4 bandas)
stack = np.dstack([blue, green, red, nir])

# Carregar polígonos de treinamento feitos no QGIS
# O arquivo deve ter um campo "classe" (int): ex 1=vegetação, 2=água, 3=solo, 4=urbano
gdf = gpd.read_file("amostras_treinamento.gpkg")
gdf = gdf.to_crs("EPSG:32721")

# Rasterizar polígonos na mesma grade da imagem
shapes = [(geom, classe) for geom, classe in zip(gdf.geometry, gdf["classe"])]
labels_raster = rasterize(
    shapes, out_shape=stack.shape[:2], transform=win_transform, fill=0, dtype="int16"
)

# Extrair amostras de treinamento
mask = labels_raster > 0
X_train = stack[mask]
y_train = labels_raster[mask]

print(f"Amostras de treinamento: {X_train.shape[0]} pixels")

# Treinar classificador Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Classificar imagem inteira
pixels = stack.reshape(-1, stack.shape[2])
y_pred = clf.predict(pixels)
classified = y_pred.reshape(stack.shape[:2])

# Exibir resultado
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Composição RGB normalizada
rgb = np.dstack([red, green, blue])
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
axes[0].imshow(rgb)
axes[0].set_title("Composição RGB")

# Mapa classificado
im = axes[1].imshow(classified, cmap="tab10")
axes[1].set_title("Classificação Supervisionada")
plt.colorbar(im, ax=axes[1], label="Classe")

plt.tight_layout()
plt.show()
