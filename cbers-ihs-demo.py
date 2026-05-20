import os

# Habilita os caches do GDAL (Opcional)
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"

import matplotlib.pyplot as plt
import numpy as np
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from skimage.transform import resize

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


# Converter para 8 bits, aplicar min-max e brilho
def normalize(array, brightness_factor):
    normalized = (array - array.min()) / (array.max() - array.min())
    return (np.clip(normalized * brightness_factor, 0, 1) * 255).astype(np.uint8)


# Obtem as imagens cortadas para cada banda
with rasterio.open(item.assets["BAND3"].href) as src:
    window = get_window(src)
    red = src.read(1, window=window)

with rasterio.open(item.assets["BAND2"].href) as src:
    green = src.read(1, window=window)

with rasterio.open(item.assets["BAND1"].href) as src:
    blue = src.read(1, window=window)

with rasterio.open(item.assets["BAND0"].href) as src:
    pan_window = get_window(src)
    pan = src.read(1, window=pan_window)

# Interpolar bandas para 4m (imita comportamento no terraview)
target_shape = (pan.shape[0] // 2, pan.shape[1] // 2)
r = resize(red.astype(float), target_shape, order=3)
g = resize(green.astype(float), target_shape, order=3)
b = resize(blue.astype(float), target_shape, order=3)
pan_resized = resize(pan.astype(float), target_shape, order=3)

# Calcular intensidade e delta para as bandas RGB
intensity = (r + g + b) / 3  # type: ignore
pan_matched = pan_resized * (intensity.mean() / pan_resized.mean())  # type: ignore
delta = pan_matched - intensity

# Aplicar delta para as bandas RGB
r_sharp = r + delta
g_sharp = g + delta
b_sharp = b + delta

# Normalizar e aplicar brilho
brightness_factor = 1.2
rgb = np.dstack(
    (
        normalize(r_sharp, brightness_factor),
        normalize(g_sharp, brightness_factor),
        normalize(b_sharp, brightness_factor),
    )
)

# Exibir resultado
plt.imshow(rgb)
plt.title("Fusão IHS CBERS4A")
plt.show()
