import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

# This single dataset file will serve both critic training scripts.

class MIMICCXRClassifierDataset(Dataset):
    """
    Dataset for the Cdiag (classification) task.
    - Loads a JPG image.
    - Converts it to RGB (as required by ResNet).
    - Returns the image and its corresponding Pneumonia label.
    """
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        dicom_id = row['dicom_id']

        # Construct the path to the JPG image
        image_path = os.path.join(
            self.image_dir, f'p{subject_id[:2]}', f'p{subject_id}', f's{study_id}', f'{dicom_id}.jpg'
        )

        # Load image and convert to a numpy array in RGB format
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Get the label
        label = torch.tensor(row['Pneumonia'], dtype=torch.float32)

        return image, label.unsqueeze(0)


class MIMICXRSegmentationDataset(Dataset):
    """
    Dataset for the Cseg (segmentation) task.
    - Loads a JPG image (as grayscale).
    - Loads its corresponding pre-generated PNG mask.
    - Returns both the image and the mask.
    """
    def __init__(self, df, image_dir, mask_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        dicom_id = row['dicom_id']

        # Construct paths for both image and mask
        image_path = os.path.join(
            self.image_dir, f'p{subject_id[:2]}', f'p{subject_id}', f's{study_id}', f'{dicom_id}.jpg'
        )
        mask_path = os.path.join(self.mask_dir, f"{dicom_id}.png")

        # Load image and mask as grayscale numpy arrays
        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Normalize mask values from [0, 255] to [0.0, 1.0]
        mask[mask == 255.0] = 1.0

        # Apply augmentations (Albumentations will transform both identically)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Add a channel dimension for the mask for consistency
        return image, mask.unsqueeze(0)