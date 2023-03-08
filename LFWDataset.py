import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os

# extract names in each row and store in a set
def extract_names(text_path):
    name_set = set()
    with open(text_path, 'r') as file:
        for line in file:           
            elements = line.split()
            if len(elements) == 3:
                name = elements[0] 
                name_set.add(name)
            elif len(elements) == 4:
                name1 = elements[0]
                name2 = elements[2]
                name_set.add(name1)
                name_set.add(name2)
    return name_set

class LFWDataset(Dataset):

    def __init__(self, dataset_path, text_path):
        super().__init__()
        self.normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.images = []
        self.targets = []

        name_set = extract_names(text_path)
        self.name_set_length = len(name_set)
        for name in name_set:
            all_images = os.listdir(os.path.join(dataset_path, name))
            for one_image in all_images:
                image_path = os.path.join(dataset_path, name, one_image)
                with open(image_path, 'rb') as f:
                    image = Image.open(f)
                    image.load()
                image_augment = image.transpose(Image.FLIP_LEFT_RIGHT) # Augmentation by horizontally flip an image
                self.images.append(image)
                self.targets.append(name)
                self.images.append(image_augment)
                self.targets.append(name)
        
        self.name_to_index = {} # map names to index
        for i, name in enumerate(name_set):
            self.name_to_index[name] = i

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        image_tensor = self.normalize(torchvision.transforms.ToTensor()(image))
        name = self.targets[idx]
        target_tensor = torch.tensor([self.name_to_index[name]]) # get the corresponding index of the name
        
        return image_tensor, target_tensor
    
class LFWDataset_test(Dataset):

    def __init__(self, dataset_path, text_path):
        super().__init__()
        self.normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        self.images1 = []
        self.images2 = []
        self.targets = []

        with open(text_path, 'r') as file:
            for line in file:           
                elements = line.split()
                image1_path = ''
                image2_path = ''
                target = True
                if len(elements) == 1:
                    continue
                if len(elements) == 3:
                    name = elements[0] 
                    all_images = os.listdir(os.path.join(dataset_path, name))
                    all_images_sorted = sorted(all_images) # sort files based on name
                    image1_idx = int(elements[1])
                    image2_idx = int(elements[2])
                    image1_path = all_images_sorted[image1_idx-1]
                    image1_path = os.path.join(dataset_path, name, image1_path)
                    image2_path = all_images_sorted[image2_idx-1]
                    image2_path = os.path.join(dataset_path, name, image2_path)
                    target = True
                elif len(elements) == 4:
                    name1 = elements[0]
                    name2 = elements[2]
                    all_images_name1 = os.listdir(os.path.join(dataset_path, name1))
                    all_images_name2 = os.listdir(os.path.join(dataset_path, name2))
                    all_images_name1_sorted = sorted(all_images_name1)
                    all_images_name2_sorted = sorted(all_images_name2)
                    image1_idx = int(elements[1])
                    image2_idx = int(elements[3])
                    image1_path = all_images_name1_sorted[image1_idx-1]
                    image1_path = os.path.join(dataset_path, name1, image1_path)
                    image2_path = all_images_name2_sorted[image2_idx-1]
                    image2_path = os.path.join(dataset_path, name2, image2_path)
                    target = False

                with open(image1_path, 'rb') as f:
                    image1 = Image.open(f)
                    image1.load()
                with open(image2_path, 'rb') as f:
                    image2 = Image.open(f)
                    image2.load()
                self.images1.append(image1)
                self.images2.append(image2)
                self.targets.append(target)

    def __len__(self):
        return len(self.images1)
        
    def __getitem__(self, idx):
        image1 = self.images1[idx]
        image2 = self.images2[idx]
        image1_tensor = self.normalize(torchvision.transforms.ToTensor()(image1))
        image2_tensor = self.normalize(torchvision.transforms.ToTensor()(image2))
        target = self.targets[idx]
        target_tensor = torch.tensor([target]) # get the corresponding index of the name
        
        return image1_tensor, image2_tensor, target_tensor