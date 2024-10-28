#!/usr/bin/env python3

from PIL import Image
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def compute_lpips(original: np.array, compressed: np.array) -> float:
  if original.shape != compressed.shape:
    raise("Original and compressed array have different shapes")

  loss_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex')
  transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

  original_transformed = transform(transforms.ToPILImage()(original)).unsqueeze(0)
  compressed_transformed = transform(transforms.ToPILImage()(compressed)).unsqueeze(0)

  distance = loss_fn(original_transformed, compressed_transformed)
  return distance.item()


def main():
  parser = argparse.ArgumentParser(prog='psnr')
  parser.add_argument("-o", "--original", help="original image")
  parser.add_argument("-c", "--compressed", help="compressed image")
  args = parser.parse_args()

  print("Original image = [{}]".format(args.original))
  print("Compressed image = [{}]".format(args.compressed))

  original_image = np.array(Image.open(args.original).convert("RGB"))
  compressed_image = np.array(Image.open(args.compressed).convert("RGB"))
  lpips_score = compute_lpips(original=original_image, compressed=compressed_image)
  print("LPIPS = [{:.2f}]".format(lpips_score))


if __name__ == "__main__":
  main()

