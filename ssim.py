#!/usr/bin/env python3

from PIL import Image
import numpy as np
from math import sqrt
import argparse

def ssim(original: np.array, compressed: np.array, k1: float = 0.01, k2: float = 0.03) -> float:
  if original.shape != compressed.shape:
    raise("Original and compressed array have different shapes")

  mu_x = np.mean(original)
  mu_y = np.mean(compressed)
  sigma_x = sqrt(np.mean((original - mu_x) ** 2))
  sigma_y = sqrt(np.mean((compressed - mu_y) ** 2))
  sigma_xy = np.mean((original - mu_x) * (compressed - mu_y))
  L = 255
  c1 = (k1 * L) ** 2.0
  c2 = (k2 * L) ** 2.0

  ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x * sigma_x + sigma_y * sigma_y + c2))
  return ssim


def main():
  parser = argparse.ArgumentParser(prog='psnr')
  parser.add_argument("-o", "--original", help="original image")
  parser.add_argument("-c", "--compressed", help="compressed image")
  args = parser.parse_args()

  print("Original image = [{}]".format(args.original))
  print("Compressed image = [{}]".format(args.compressed))

  original_image = np.array(Image.open(args.original).convert("RGB"))
  compressed_image = np.array(Image.open(args.compressed).convert("RGB"))
  ssim_value = ssim(original=original_image, compressed=compressed_image)
  print("SSIM = [{:.2f}]".format(ssim_value))


if __name__ == "__main__":
  main()

