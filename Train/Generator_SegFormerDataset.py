import pytorch_lightning as pl
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import random
import pandas as pd


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_segmentation_map(mask_path):
    mask = np.array(Image.open(mask_path))
    # Remove the singleton dimension
    mask = mask.squeeze()
    return mask



class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir,csvFile, feature_extractor):
      """
      Args:
          root_dir (string): Root directory of the dataset containing the CSV file, images, and annotations.
          feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
      """
      self.root_dir = root_dir
      self.feature_extractor = feature_extractor

      # Read the CSV file containing filenames
      csv_file = os.path.join(self.root_dir,csvFile)
      filenames_df = pd.read_csv(csv_file)
      filenames = filenames_df["filename"].tolist()

      self.images = []
      self.masks = []

      # Iterate over filenames in the CSV
      for filename in filenames:
          # Image path
          image_path = os.path.join(self.root_dir, "image", f"{filename}.png")
          if os.path.exists(image_path):
              self.images.append(image_path)

          # Label path
          label_path = os.path.join(self.root_dir, "label", f"{filename}")
          if os.path.exists(label_path):
              self.masks.append(label_path)

      # Ensure images and masks have the same length
      print(f"Images loaded: {len(self.images)}")
      assert len(self.images) == len(self.masks), "Number of images and masks should be equal"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images[idx])
        mask_path = os.path.join(self.root_dir, self.masks[idx])

        image = preprocess_image(image_path)
        segmentation_map = preprocess_segmentation_map(mask_path)

        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs
