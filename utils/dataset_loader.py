import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class QRDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []

        for label, folder in enumerate(["original", "fake"]):
            folder_path = os.path.join(root_dir, folder)
            for img_name in os.listdir(folder_path):
                self.images.append(os.path.join(folder_path, img_name))
                self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        img = self.transform(img)

        return img, label
