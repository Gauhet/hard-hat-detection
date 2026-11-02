"""Find near-duplicate photos, so that no photo (or a resized copy of it) ends up in both the
training data and a test set.

Each photo is reduced to a 64-bit difference hash: shrink it to a 9x8 grey grid (ignoring its
aspect ratio, since one source stretched every photo to a square) and record whether each pixel
is brighter than its left neighbour. Copies of the same photo give hashes a few bits apart.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

# Number of set bits in each byte value, for counting differing bits quickly.
BIT_COUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: str | Path) -> int:
    with Image.open(path) as image:
        grid = np.asarray(image.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (grid[:, 1:] > grid[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def hash_all(paths: list[Path], workers: int | None = None) -> np.ndarray:
    with ProcessPoolExecutor(workers) as pool:
        return np.array(list(pool.map(dhash, paths, chunksize=64)), dtype=np.uint64)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise number of differing bits between two arrays of 64-bit hashes."""
    xor = a[:, None] ^ b[None, :]
    return BIT_COUNT[xor.view(np.uint8).reshape(*xor.shape, 8)].sum(axis=-1)


def group_duplicates(
    hashes: np.ndarray, max_bits: int = 10, sources: list[str | None] | None = None, chunk: int = 512
) -> np.ndarray:
    """Label each photo with a group id. Photos share a group if their hashes differ by at most
    `max_bits`, or if they have the same non-empty `sources` entry (frames of one video),
    directly or through a chain of such links."""
    parent = np.arange(len(hashes))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(hashes), chunk):
        close = hamming(hashes[start:start + chunk], hashes) <= max_bits
        for i, j in zip(*np.nonzero(close)):
            a, b = root(start + int(i)), root(int(j))
            if a != b:
                parent[max(a, b)] = min(a, b)
    first_of_source: dict[str, int] = {}
    for i, source in enumerate(sources or []):
        if source:
            a, b = root(first_of_source.setdefault(source, i)), root(i)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return np.array([root(i) for i in range(len(hashes))])
