"""The shared, metadata-independent image decoder."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import math

import numpy as np
from PIL import Image, ImageOps


@dataclass
class DecodeResult:
    rgb: np.ndarray | None
    status: str
    original_width: int = 0
    original_height: int = 0
    oriented_width: int = 0
    oriented_height: int = 0
    input_mode: str = ""
    alpha_composited: bool = False
    exif_transposed: bool = False


def decode_image(source: str | Path | bytes) -> DecodeResult:
    try:
        with Image.open(BytesIO(source) if isinstance(source, bytes) else source) as im:
            im.load()
            width, height, mode = im.width, im.height, im.mode
            base = dict(original_width=width, original_height=height, input_mode=mode)
            if width <= 0 or height <= 0:
                return DecodeResult(None, "decode_error", **base)
            if (im.format not in {"JPEG", "PNG"} or getattr(im, "n_frames", 1) > 1
                    or mode in {"I", "I;16", "I;16B", "I;16L", "F"} or mode.startswith("I;16")):
                return DecodeResult(None, "unsupported_format", **base)
            orientation = im.getexif().get(274, 1)
            oriented = ImageOps.exif_transpose(im)
            alpha = "A" in oriented.getbands() or "transparency" in oriented.info
            if alpha:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                rgb = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                rgb = oriented.convert("RGB")
            arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
            return DecodeResult(arr, "ok", oriented_width=rgb.width, oriented_height=rgb.height,
                                alpha_composited=alpha, exif_transposed=orientation != 1, **base)
    except (OSError, ValueError, SyntaxError):
        return DecodeResult(None, "decode_error")


def canonical_content_id(rgb_uint8: np.ndarray) -> str:
    x = np.asarray(rgb_uint8)
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("canonical content requires uint8 RGB")
    h, w, _ = x.shape
    digest = hashlib.sha256()
    digest.update(h.to_bytes(8, "little"))
    digest.update(w.to_bytes(8, "little"))
    digest.update(np.ascontiguousarray(x).tobytes(order="C"))
    return digest.hexdigest()


def resize_if_needed(rgb_uint8: np.ndarray) -> tuple[np.ndarray, dict]:
    x = np.asarray(rgb_uint8)
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("resize requires uint8 RGB")
    h, w = x.shape[:2]
    if min(h, w) <= 0:
        raise ValueError("empty image")
    if min(h, w) >= 64:
        out = np.ascontiguousarray(x)
    elif h <= w:
        out = np.asarray(Image.fromarray(x).resize((math.ceil(64 * w / h), 64), Image.Resampling.BILINEAR))
    else:
        out = np.asarray(Image.fromarray(x).resize((64, math.ceil(64 * h / w)), Image.Resampling.BILINEAR))
    oh, ow = out.shape[:2]
    return np.ascontiguousarray(out), dict(processed_width=ow, processed_height=oh,
                                             upscaled=min(h, w) < 64, scale_x=ow / w, scale_y=oh / h)
