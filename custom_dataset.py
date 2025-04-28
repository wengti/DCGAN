from pathlib import Path
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import random
import torch

def find_classes(directory):
    directory = Path(directory)
    
    class_names = list(entry.name for entry in os.scandir(directory))
    if not class_names:
        raise FileNotFoundError(f"[WARNING] No valid class names can be found in {directory}. Please check.")
    
    class_to_label = {}
    for idx, name in enumerate(class_names):
        class_to_label[name] = idx
    
    return class_names, class_to_label

class MNIST_dataset(Dataset):
    
    def __init__(self, directory, config):
        directory = Path(directory)
        self.classes, self.class_to_label = find_classes(directory)
        self.path_list = list(directory.glob("*/*.png"))
        self.config = config
    
    def load_image(self, index):
        img = Image.open(self.path_list[index])
        return img
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        im_size = self.config['im_size']
        img = self.load_image(index)
        simple_transform = transforms.Compose([transforms.ToTensor(),
                                               transforms.Resize((im_size,im_size))])
        img = simple_transform(img) # (CxHxW), 0 to 1, float tensor
        
        if self.config['im_channels'] == 3:
            r = img * random.uniform(0.2, 1.0)
            g = img * random.uniform(0.2, 1.0)
            b = img * random.uniform(0.2, 1.0)
            img = torch.cat([r,g,b], dim=0)
        
        img = (img*2) - 1 # (CxHxW), -1 to 1, float tensor
    
        return img


class Celeb_dataset(Dataset):
    def __init__(self, directory, config):
        directory = Path(directory)
        self.config = config
        self.path_list = list(directory.glob('*/*.jpg'))
    
    def load_image(self, index):
        img = Image.open(self.path_list[index])
        return img
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        im_size = self.config['im_size']
        img = self.load_image(index)
        simpleTransform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((im_size, im_size)),
                                              transforms.CenterCrop((im_size, im_size))])
        img = simpleTransform(img) # (CxHxW), 0 to 1, float tensor
        img = (img*2) - 1 # (CxHxW), 0 to 1, float tensor
        img = torch.clamp(img, -1, 1)
        
        return img
        
        
    