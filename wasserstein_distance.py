#!/usr/bin/env python3

from PIL import Image
import numpy as np
from math import sqrt
import argparse

def wasserstein_distance(original: np.array, compressed: np.array) -> float:
  if original.shape != compressed.shape:
    raise("Original and compressed array have different shapes")

  flattened_original = original.flatten()
  flattened_compressed = compressed.flatten()

  sorted_original = np.sort(flattened_original)
  sorted_compressed = np.sort(flattened_compressed)
  # sorted_original = flattened_original
  # sorted_compressed = flattened_compressed

  commulative_difference = np.sum(np.abs(sorted_original - sorted_compressed))
  return commulative_difference / flattened_original.size


def main():
  parser = argparse.ArgumentParser(prog='psnr')
  parser.add_argument("-o", "--original", help="original image")
  parser.add_argument("-c", "--compressed", help="compressed image")
  args = parser.parse_args()

  print("Original image = [{}]".format(args.original))
  print("Compressed image = [{}]".format(args.compressed))

  original_image = np.array(Image.open(args.original).convert("RGB"))
  compressed_image = np.array(Image.open(args.compressed).convert("RGB"))
  wasserstein_value = wasserstein_distance(original=original_image, compressed=compressed_image)
  print("Wasserstein distance = [{:.2f}]".format(wasserstein_value))


if __name__ == "__main__":
  main()

