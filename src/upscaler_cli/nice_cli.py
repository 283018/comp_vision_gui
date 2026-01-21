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

# just stfu pls

import argparse
from pathlib import Path

from upscaler.image_api import Upscaler, load_image, save_image


def main():
    parser = argparse.ArgumentParser(description="Upscale an image using SRGAN.")
    parser.add_argument("input", type=Path, help="Path to input image")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output.png"), help="Output path (default: output.png)",
    )
    parser.add_argument("-p", "--patch-size", type=int, default=128, help="Patch size (default: 128)")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between patches (default: 0)")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Upscaling factor (default: 4)")

    args = parser.parse_args()

    upscaler = Upscaler()
    upscaler.load()
    img = load_image(args.input)
    sr = upscaler.upscale(img, patch_size=args.patch_size, overlap=args.overlap, scale=args.scale)
    out_path = Path(f"upscaled_{args.scale}_{args.output}").resolve()
    save_image(sr, out_path)
    
    
    print(f"\nImage upscaled and saved into:\n{out_path}")


if __name__ == "__main__":
    main()
