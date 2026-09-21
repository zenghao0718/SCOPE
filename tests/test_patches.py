import numpy as np
from data.patches import patch_coordinates, extract_patches


def test_coordinates_and_duplicates():
    assert [patch_coordinates(x, x)[0, 0] for x in (64, 80, 128, 256)] == [0, 0, 0, 32]
    assert [patch_coordinates(x, x)[-1, 0] for x in (64, 80, 128, 256)] == [0, 16, 64, 160]
    patches, coords, unique = extract_patches(np.zeros((64, 64, 3)))
    assert patches.shape == (4, 64, 64, 3) and unique == 1
