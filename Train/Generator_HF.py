import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import SegformerFeatureExtractor

class SegformerDataset(Dataset):  # Changed class name to reflect PyTorch Dataset
    def __init__(self, csv_filename, image_folder, label_folder, feature_extractor, input_size=(512, 512), augmentation=None):
        self.csv_filename = csv_filename
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.feature_extractor = feature_extractor
        self.input_size = input_size
        self.augmentation = augmentation
        self.data = pd.read_csv(self.csv_filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.data["image"][idx])
        label_path = os.path.join(self.label_folder, self.data["label"][idx])
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # Assuming grayscale labels

        if self.augmentation:
            image, label = self.augmentation(image, label)

        image = image.resize(self.input_size)
        label = label.resize(self.input_size)

        encoded_inputs = self.feature_extractor(images=image, labels=label)
        return encoded_inputs
