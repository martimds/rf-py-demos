# type: ignore
import asyncio
import struct
import time
from pathlib import Path

import mlx.core as mx
from async_tiff import TIFF
from async_tiff.store import LocalStore

INPUT = Path("imagens/sentinel2_2020_321.tiff")
OUTPUT = Path("imagens/sentinel2_2020_321_mlx.tiff")
SCALE = 1.5


async def read_geotiff(path):
    store = LocalStore(str(path.parent))
    tiff = await TIFF.open(path.name, store=store)
    image = tiff.ifds[0]

    width, height = image.image_width, image.image_height
    tile_width, tile_height = image.tile_width, image.tile_height
    tiles_x, tiles_y = image.tile_count

    positions = [(x, y) for y in range(tiles_y) for x in range(tiles_x)]
    raw_tiles = await image.fetch_tiles(positions)
    decoded_tiles = await asyncio.gather(*(t.decode() for t in raw_tiles))

    rows = []
    index = 0
    for row_idx in range(tiles_y):
        row_parts = []
        for col_idx in range(tiles_x):
            tile = mx.array(decoded_tiles[index])
            crop_h = min(tile_height, height - row_idx * tile_height)
            crop_w = min(tile_width, width - col_idx * tile_width)
            if tile.shape[0] != crop_h or tile.shape[1] != crop_w:
                tile = tile[:crop_h, :crop_w]
            row_parts.append(tile)
            index += 1
        rows.append(mx.concatenate(row_parts, axis=1))

    return mx.concatenate(rows, axis=0)


def scale_image(image, factor):
    return (image.astype(mx.float32) * factor).astype(image.dtype)


def write_tiff(image, path):
    mx.eval(image)
    pixel_data = bytes(memoryview(image))

    height, width = image.shape[0], image.shape[1]
    bands = image.shape[2] if image.ndim > 2 else 1

    sample_format, bits = {
        mx.uint8: (1, 8),
        mx.int16: (2, 16),
        mx.uint16: (1, 16),
        mx.int32: (2, 32),
        mx.uint32: (1, 32),
        mx.float32: (3, 32),
    }[image.dtype]

    tags = [
        (256, 3, 1, width),
        (257, 3, 1, height),
        (258, 3, bands, bits),
        (259, 3, 1, 1),
        (262, 3, 1, 2 if bands >= 3 else 1),
        (273, 4, 1, 0),
        (277, 3, 1, bands),
        (278, 4, 1, height),
        (279, 4, 1, len(pixel_data)),
        (284, 3, 1, 1),
        (339, 3, 1, sample_format),
    ]

    ifd_size = 2 + len(tags) * 12 + 4
    extra = bands * 2 if bands > 1 else 0
    data_start = 8 + ifd_size + extra
    bits_offset = 8 + ifd_size

    with open(path, "wb") as f:
        f.write(struct.pack("<2sHI", b"II", 42, 8))
        f.write(struct.pack("<H", len(tags)))

        for tag, typ, count, value in tags:
            if tag == 273:
                f.write(struct.pack("<HHII", tag, typ, count, data_start))
            elif tag == 258 and bands > 1:
                f.write(struct.pack("<HHII", tag, 3, count, bits_offset))
            else:
                f.write(struct.pack("<HHII", tag, typ, count, value))

        f.write(struct.pack("<I", 0))

        if bands > 1:
            f.write(struct.pack(f"<{bands}H", *([bits] * bands)))

        f.write(pixel_data)


async def main():
    t0 = time.perf_counter()

    image = await read_geotiff(INPUT)
    result = scale_image(image, SCALE)
    write_tiff(result, OUTPUT)

    print(f"Concluído {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
