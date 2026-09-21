from io import BytesIO
import numpy as np
from PIL import Image, PngImagePlugin
from data.decoding import decode_image, canonical_content_id, resize_if_needed


def encoded(image, fmt="PNG", **kw):
    stream = BytesIO()
    image.save(stream, format=fmt, **kw)
    return stream.getvalue()


def test_rgb_gray_alpha_and_content_id():
    x = Image.fromarray(np.full((4, 5, 3), 128, np.uint8))
    png = encoded(x)
    jpeg = encoded(x, "JPEG")
    assert decode_image(png).rgb.shape == (4, 5, 3)
    assert decode_image(jpeg).rgb.dtype == np.uint8
    gray = decode_image(encoded(Image.new("L", (4, 5), 77))).rgb
    assert np.all(gray == 77)
    rgba = decode_image(encoded(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))).rgb
    assert rgba.tolist() == [[[255, 255, 255]]]
    meta = PngImagePlugin.PngInfo()
    meta.add_text("irrelevant", "metadata")
    assert canonical_content_id(decode_image(png).rgb) == canonical_content_id(decode_image(encoded(x, pnginfo=meta)).rgb)


def test_unsupported_and_corrupt():
    sixteen = encoded(Image.fromarray(np.full((3, 3), 1024, np.uint16)))
    assert decode_image(sixteen).status == "unsupported_format"
    assert decode_image(encoded(Image.new("F", (2, 2)), "TIFF")).status == "unsupported_format"
    first, second = Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2), "white")
    stream = BytesIO()
    first.save(stream, format="GIF", save_all=True, append_images=[second])
    assert decode_image(stream.getvalue()).status == "unsupported_format"
    assert decode_image(b"not an image").status == "decode_error"


def test_resize_exact_ceil():
    out, info = resize_if_needed(np.zeros((20, 41, 3), np.uint8))
    assert out.shape == (64, 132, 3)
    assert info["upscaled"]


def test_exif_orientation():
    image = Image.new("RGB", (2, 3), "red")
    exif = Image.Exif()
    exif[274] = 6
    result = decode_image(encoded(image, "JPEG", exif=exif))
    assert result.exif_transposed and result.rgb.shape == (2, 3, 3)
