import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import median_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from yaspin import yaspin

RASTER_PATH = "IMAGES/sentinel2_2020_indices.tiff"
SAMPLES_PATH = "SAMPLES/sentinel2020.gpkg"
OUTPUT_PATH = "CLASSIFIED/sentinel2_2020_classified.tiff"

CLASS_NAMES = {
    1: "vegetação",
    2: "urbano",
    3: "água",
    4: "plantio",
    5: "plantio 2",
    6: "solo exposto",
}

os.makedirs("CLASSIFIED", exist_ok=True)

with yaspin(text="Lendo raster") as sp:
    with rasterio.open(RASTER_PATH) as src:
        img = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
    sp.ok("✓")

n_bands, height, width = img.shape

with yaspin(text="Lendo amostras") as sp:
    gdf = gpd.read_file(SAMPLES_PATH).to_crs(profile["crs"])
    shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf.uso_solo)]
    labels = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    sp.ok("✓")

mask = labels > 0  # type: ignore
X_train = img[:, mask].T
y_train = labels[mask] - 1  # type: ignore

MAX_SAMPLES = 200_000
if len(X_train) > MAX_SAMPLES:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), MAX_SAMPLES, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]

print(f"Numero de pixels: {len(y_train)}")
for cls, name in CLASS_NAMES.items():
    print(f"{cls} ({name}): {(y_train == cls - 1).sum()}")

clf = HistGradientBoostingClassifier(
    max_iter=300,
    max_depth=10,
    max_leaf_nodes=127,
    learning_rate=0.1,
    min_samples_leaf=10,
    l2_regularization=0.1,
    random_state=42,
)

with yaspin(text="Treinando classificador") as sp:
    clf.fit(X_train, y_train)
    sp.ok("✓")

y_pred = clf.predict(X_train)

with yaspin(text="Classificando imagem") as sp:
    pixels = img.reshape(n_bands, -1).T
    prediction = (clf.predict(pixels) + 1).astype(np.uint8).reshape(height, width)
    sp.ok("✓")

with yaspin(text="Filtrando ruído (median 3x3)") as sp:
    prediction = median_filter(prediction, size=3).astype(np.uint8)
    sp.ok("✓")

with yaspin(text="Salvando resultado") as sp:
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
        dst.write(prediction, 1)
    sp.ok("✓")

print(f"Concluído: {OUTPUT_PATH}")
print(f"Precisão: {np.mean(y_pred == y_train):.3f}")
print(
    classification_report(
        y_train, y_pred, target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    )
)
