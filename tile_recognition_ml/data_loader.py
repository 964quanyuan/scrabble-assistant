import os
from PIL import Image  # Assuming you use Pillow (PIL Fork) for image loading
import torch.utils.data as data
import numpy as np
import torch

class ImageFolderDataset(data.Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        self.data = []  # List to store image paths and labels
        self.dict = {chr(i): i - 65 for i in range(65, 91)}
        self.dict["false+_reg"] = 27
        self.dict["false+_mult"] = 28
        self.dict["false+_star"] = 29
        self.dict["blank"] = 26

        # Construct data list based on train/test split
        data_dir = os.path.join(self.root_dir, "train" if self.train else "test")
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    label = class_name  # Assuming class name is the label
                    self.data.append((image_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = self.load_image(image_path)
        label = self.dict[label]
        label = torch.LongTensor([label])
        return image, label

    def load_image(self, image_path):
        # Load grayscale image using Pillow
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        # Convert to tensor (assuming float for normalization later)
        image = torch.from_numpy(np.array(image, dtype=np.float32))
        # Normalize (optional): You can add normalization here if needed
        image = image / 255.0  # Normalize to 0-1 range (example)
        return image
    
