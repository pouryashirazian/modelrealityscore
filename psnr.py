#!/usr/bin/env python3

from PIL import Image
import numpy as np
from math import log10, sqrt
import argparse
import os

def psnr(original: np.array, compressed: np.array) -> float:
  if original.shape != compressed.shape:
    raise("Original and compressed array have different shapes")

  mse = np.mean((original - compressed) ** 2)
  if mse == 0.0:
    return 1.0

  max_intensity = 255.0
  return 20.0 * log10(max_intensity / sqrt(mse))


def main():
  parser = argparse.ArgumentParser(prog='psnr')
  parser.add_argument("-o", "--original", help="original image")
  parser.add_argument("-c", "--compressed", help="compressed image")
  args = parser.parse_args()

  print("Original image = [{}]".format(args.original))
  print("Compressed image = [{}]".format(args.compressed))

  original_image = np.array(Image.open(args.original).convert("RGB"))
  compressed_image = np.array(Image.open(args.compressed).convert("RGB"))
  psnr_value = psnr(original=original_image, compressed=compressed_image)
  print("PSNR = [{:.2f} dB]".format(psnr_value))


if __name__ == "__main__":
  main()

