import numpy as np
from PIL import Image

from hardhat_detector.dedupe import dhash, group_duplicates, hamming


def photo(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((480, 360), Image.BILINEAR)


def test_resized_and_stretched_copies_hash_alike_and_other_photos_do_not(tmp_path):
    original, other = photo(0), photo(1)
    paths = {}
    for name, image in {
        "original": original,
        "square": original.resize((640, 640)),  # stretched, as Roboflow did
        "small": original.resize((160, 120)),
        "other": other,
    }.items():
        paths[name] = tmp_path / f"{name}.jpg"
        image.save(paths[name], quality=85)
    h = {name: np.array([dhash(p)], dtype=np.uint64) for name, p in paths.items()}
    assert hamming(h["original"], h["square"])[0, 0] <= 6
    assert hamming(h["original"], h["small"])[0, 0] <= 6
    assert hamming(h["original"], h["other"])[0, 0] > 16


def test_grouping_follows_chains_of_close_hashes():
    hashes = np.array([0b0, 0b1, 0b11, (1 << 64) - 1], dtype=np.uint64)
    groups = group_duplicates(hashes, max_bits=1)
    # 0 -> 1 -> 3 differ by one bit at each step; the last hash differs in every bit.
    assert groups[0] == groups[1] == groups[2] != groups[3]


def test_frames_of_the_same_video_are_grouped_whatever_their_hashes():
    hashes = np.array([0, (1 << 64) - 1, 0xF0F0F0F0F0F0F0F0], dtype=np.uint64)
    groups = group_duplicates(hashes, max_bits=0, sources=["clip_mp4", "clip_mp4", None])
    assert groups[0] == groups[1] != groups[2]
