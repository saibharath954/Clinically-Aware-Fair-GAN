import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pydicom
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MIMIC_CXR_Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, img_size=256):
        """
        Args:
            root_dir (string): Directory with all the DICOM images
            csv_file (string): Path to the csv file with annotations
            transform (callable, optional): Optional transform to be applied
            img_size (int): Target image size for resizing
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        
        # Precompute paths
        self.image_paths = []
        for idx in range(len(self.df)):
            subject_id = str(self.df.iloc[idx]['subject_id'])
            study_id = str(self.df.iloc[idx]['study_id'])
            dicom_id = str(self.df.iloc[idx]['dicom_id'])
            path = os.path.join(root_dir, f'p{subject_id[:2]}', f'p{subject_id}', f's{study_id}', f'{dicom_id}.dcm')
            self.image_paths.append(path)
            
        # Convert labels to numerical values
        self.labels = self.df['Pneumonia'].astype(int).values
        self.protected_attrs = pd.get_dummies(self.df['Race']).values  # One-hot encoding
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load DICOM image
        dicom_path = self.image_paths[idx]
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        
        # Convert to PIL Image and resize
        image = Image.fromarray(image).convert('L')  # Convert to grayscale
        image = image.resize((self.img_size, self.img_size))
        
        # Normalize to [-1, 1] range
        image = np.array(image, dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min()) * 2 - 1
        
        # Convert to 3-channel (for ResNet compatibility)
        image = np.stack([image]*3, axis=0)
        
        label = self.labels[idx]
        protected_attr = self.protected_attrs[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'protected_attr': torch.tensor(protected_attr, dtype=torch.float32)
        }

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.ToTensor()
])

# Example usage
if __name__ == '__main__':
    dataset = MIMIC_CXR_Dataset(
        root_dir='/path/to/mimic-cxr-jpg/2.0.0/files',
        csv_file='/path/to/mimic-cxr-2.0.0-chexpert.csv',
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Visualize first batch
    batch = next(iter(dataloader))
    images = batch['image']
    labels = batch['label']
    protected = batch['protected_attr']
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5, cmap='gray')
        ax.set_title(f"Label: {labels[i].item()}\nRace: {protected[i].argmax().item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()