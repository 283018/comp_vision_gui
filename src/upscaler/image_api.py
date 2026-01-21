import logging
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import absl.logging

absl.logging.set_verbosity(absl.logging.FATAL)

warnings.filterwarnings('ignore')

import absl.logging

absl.logging.set_verbosity('fatal')
absl.logging.use_absl_handler = False

import tensorflow as tf

tf.get_logger().setLevel(logging.FATAL)

import contextlib
import importlib.resources
from pathlib import Path
from typing import Literal

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from .model_parts import PixelShuffle


def get_weight_file(custom_path=None):
    if custom_path is not None:
        return Path(custom_path)
    try:
        traversable = importlib.resources.files("upscaler.weights") / "srgan.keras"
        return Path(str(traversable))
        
            
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        return Path(__file__).parent / "weights" / "srgan.keras"


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        img.save(path)


def pil_to_tensor(img: Image.Image) -> tf.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return tf.convert_to_tensor(arr, dtype=tf.float32)


def pil_to_float32_tensor(img: Image.Image) -> tf.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return tf.convert_to_tensor(arr, dtype=tf.float32)


# tf types are mess :/
def float32_tensor_to_uint8_array(t: tf.Tensor) -> np.ndarray:
    t = tf.clip_by_value(t, 0.0, 1.0)  # type: ignore
    return (t.numpy() * 255.0).astype(np.uint8)  # type: ignore

def _downscale_pil(img: Image.Image, factor: float) -> Image.Image:
    if factor == 1.0:
        return img
    w, h = img.size
    new_w = round(w * factor)
    new_h = round(h * factor)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _reflect_pad_array(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h < 0 or pad_w < 0:
        msg = "target smaller than arr"
        raise ValueError(msg)

    pad_top, pad_bottom = 0, pad_h
    pad_left, pad_right = 0, pad_w
    if pad_bottom == 0 and pad_right == 0:
        return arr
    return np.pad(
        arr,
        (
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0),
        ),
        mode="reflect",
    )


def split_into_patches(
    img: Image.Image,
    patch_size: int,
) -> tuple[list[Image.Image], tuple[int, int], tuple[int, int]]:
    w, h = img.size  # AGH
    arr = np.asarray(img).astype(np.uint8)  # h,w,3

    nx = (w + patch_size - 1) // patch_size
    ny = (h + patch_size - 1) // patch_size
    pad_w = nx * patch_size
    pad_h = ny * patch_size

    padded = _reflect_pad_array(arr, pad_h, pad_w)  # padded_h, padded_w, 3

    patches: list[Image.Image] = []
    for y in range(ny):
        for x in range(nx):
            y0 = y * patch_size
            x0 = x * patch_size
            tile = padded[y0 : y0 + patch_size, x0 : x0 + patch_size]
            patches.append(Image.fromarray(tile))
    return patches, (w, h), (nx, ny)


def merge_patches(  # noqa: PLR0913
    sr_patches: list[Image.Image],
    original_size: tuple[int, int],
    grid: tuple[int, int],
    patch_size: int,
    upscale: int,
    *,
    overlap: int = 0,
    blend: bool = False,
) -> Image.Image:
    """
    :param sr_patches: list of PIL patches (row-by y,x)
    :param original_size: (w, h) of original image
    :param grid: (nx, ny) number of patches per row/col
    :param patch_size: lr patch size used when splitting
    :param upscale: upscale factor
    :param overlap: number of pixels overlapped in lr coordinates. Default: 0
    :param blend: whether to perform linear alpha blending on overlaps. Default: False
    :returns: merged image
    """
    w, h = original_size
    nx, ny = grid
    sr_patch_size = patch_size * upscale
    out_w = nx * sr_patch_size
    out_h = ny * sr_patch_size

    weight = np.array(())  # to bound object
    if blend and overlap > 0:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight = np.zeros((out_h, out_w, 1), dtype=np.float32)
    else:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    i = 0
    for y in range(ny):
        for x in range(nx):
            sr_img = sr_patches[i]
            sr_arr = np.asarray(sr_img).astype(np.float32 if blend else np.uint8)
            y0 = y * sr_patch_size
            x0 = x * sr_patch_size

            if blend and overlap > 0:
                ph, pw = sr_arr.shape[:2]
                alpha = np.ones((ph, pw, 1), dtype=np.float32)

                ov_px = overlap * upscale
                if ov_px > 0:
                    ramp = np.linspace(0, 1, ov_px, endpoint=False).reshape(1, ov_px, 1)
                    if x > 0:
                        alpha[:, :ov_px, :] *= ramp
                    if x < nx - 1:
                        alpha[:, -ov_px:, :] *= ramp[:, ::-1, :]

                    ramp_v = np.linspace(0, 1, ov_px, endpoint=False).reshape(ov_px, 1, 1)
                    if y > 0:
                        alpha[:ov_px, :, :] *= ramp_v
                    if y < ny - 1:
                        alpha[-ov_px:, :, :] *= ramp_v[::-1, ...]

                canvas[y0 : y0 + ph, x0 : x0 + pw, :] += sr_arr.astype(np.float32) * alpha
                weight[y0 : y0 + ph, x0 : x0 + pw, :] += alpha
            else:
                canvas[y0 : y0 + sr_arr.shape[0], x0 : x0 + sr_arr.shape[1], :] = sr_arr

            i += 1

    if blend and overlap > 0:
        weight[weight == 0] = 1.0
        canvas = (canvas / weight).astype(np.uint8)
    elif canvas.dtype != np.uint8:
        canvas = canvas.astype(np.uint8)

    final = canvas[: h * upscale, : w * upscale, :]
    return Image.fromarray(final)


class Upscaler:
    generator: keras.Model | None = None
    _NATIVE_SCALE: int = 4
    _ALLOWED_SCALES: tuple[int, ...] = (2, 4, 8, 16)
    
    def load(self, path: Path | None = None):
        path = get_weight_file(path)
        if not path.exists():
            msg = f"Serialized model weights are not found in {path}"
            raise RuntimeError(msg)

        self.generator = keras.models.load_model(
            path.resolve(),
            custom_objects={"PixelShuffle": PixelShuffle},
            compile=False,
        )

    def _upscale_tensor(self, patch: tf.Tensor) -> tf.Tensor:
        if self.generator is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        lr = tf.expand_dims(patch, axis=0)
        sr = self.generator(lr, training=False)
        sr = tf.squeeze(sr, axis=0)
        return tf.clip_by_value(sr, 0.0, 1.0)  # type: ignore

    def _upscale_patches_batch(
        self,
        patches: list[Image.Image],
        batch_size: int = 8,
    ) -> list[Image.Image]:
        sr_patches: list[Image.Image] = []
        if self.generator is None:
            msg = "First load model!"
            raise RuntimeError(msg)

        n = len(patches)
        i = 0
        while i < n:
            batch = patches[i : i + batch_size]
            # stack tensors
            t_batch = np.stack([np.asarray(p).astype(np.float32) / 255.0 for p in batch], axis=0)  # N,H,W,3
            t_batch_tf = tf.convert_to_tensor(t_batch, dtype=tf.float32)
            sr_batch_tf = self.generator(t_batch_tf, training=False)  # N, H*up, W*up, 3
            for j in range(sr_batch_tf.shape[0]):
                arr = float32_tensor_to_uint8_array(sr_batch_tf[j])
                sr_patches.append(Image.fromarray(arr))
            i += batch_size
        return sr_patches

    def _upscale_x4(
        self,
        img: Image.Image,
        *,
        patch_size: int,
        batch_size: int,
        overlap: int,
        blend: bool,
    ) -> Image.Image:
        patches, original_size, grid = split_into_patches(img, patch_size)
        sr_patches = self._upscale_patches_batch(patches, batch_size=batch_size)

        return merge_patches(
            sr_patches,
            original_size,
            grid,
            patch_size,
            upscale=self._NATIVE_SCALE,
            overlap=overlap,
            blend=blend,
        )
    
    def upscale(  # noqa: PLR0913
        self,
        img: Image.Image,
        *,
        scale: Literal[2, 4, 8, 16, 32] = 4 ,
        patch_size: int = 128,
        batch_size: int = 8,
        overlap: int = 0,
        blend: bool = False,
    ) -> Image.Image:
        """
        :param img: lr input (HxW)
        :param patch_size: patch size used for splitting
        :param overlap: number of pixels overlapped between patches (for blending)
        :param blend: enable linear blending in overlaps
        :returns: upscaled PIL image
        """
        if self.generator is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)
        if patch_size < 1:
            msg = "patch_size must be >=1"
            raise ValueError(msg)

        if scale not in self._ALLOWED_SCALES:
            msg = f"scale must be one of {self._ALLOWED_SCALES}"
            raise ValueError(msg)

        out = img

        if scale == 2:
            out = self._upscale_x4(
                out,
                patch_size=patch_size,
                batch_size=batch_size,
                overlap=overlap,
                blend=blend,
            )
            out = _downscale_pil(out, 0.5)

        elif scale == 4:
            out = self._upscale_x4(
                out,
                patch_size=patch_size,
                batch_size=batch_size,
                overlap=overlap,
                blend=blend,
            )

        elif scale == 8:
            for _ in range(2):
                out = self._upscale_x4(
                    out,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    overlap=overlap,
                    blend=blend,
                )

        elif scale == 16:
            for _ in range(3):
                out = self._upscale_x4(
                    out,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    overlap=overlap,
                    blend=blend,
                )
        elif scale == 32:
            for _ in range(4):
                out = self._upscale_x4(
                    out,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    overlap=overlap,
                    blend=blend,
                )

        return out

    def __call__(self, img: Image.Image, **kwargs) -> Image.Image:
        return self.upscale(img, **kwargs)
