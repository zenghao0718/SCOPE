"""Four ordered deterministic crops."""
import numpy as np


def patch_coordinates(height: int, width: int) -> np.ndarray:
    if min(height, width) < 64:
        raise ValueError("resize before patch extraction")
    def ends(length):
        return (int(np.clip(length // 4 - 32, 0, length - 64)),
                int(np.clip(3 * length // 4 - 32, 0, length - 64)))
    ay, by = ends(height)
    ax, bx = ends(width)
    return np.array([[ay, ax], [ay, bx], [by, ax], [by, bx]], dtype=np.int32)


def extract_patches(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    coords = patch_coordinates(*image.shape[:2])
    patches = np.stack([image[y:y+64, x:x+64, :] for y, x in coords])
    return patches, coords, len({np.ascontiguousarray(p).tobytes() for p in patches})
