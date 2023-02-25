import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import glob
import random


class TrainDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None, simple_transform=None, hard_transform=None, train_idxs=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.train_idxs = train_idxs
        self.simple_transform = simple_transform
        self.hard_transform = hard_transform

    def __len__(self):
        if self.train_idxs is not None:
            return len(self.train_idxs)
        else:
            return len(self.img_labels)

    def __getitem__(self, idx):
        if self.train_idxs is not None:
            img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[self.train_idxs[idx], 0]))
            label = self.img_labels.iloc[self.train_idxs[idx], 1]
        else:
            img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]))
            label = self.img_labels.iloc[idx, 1]
        image = read_image(img_path + '.jpg') # read img

        # apply different data transform based on different probs.
        if self.transform and self.simple_transform and self.hard_transform:
            ra = random.random()
            if ra < 0.3:
                image = self.transform(image)
            elif ra < 0.6:
                image = self.simple_transform(image)
            else:
                image = self.hard_transform(image)

        elif self.transform and self.simple_transform:
            if random.random() > 0.5:
                image = self.transform(image)
            else:
                image = self.simple_transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, label

# This class of dataset is for the Validation step
class ValDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None, val_idxs=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.val_idxs = val_idxs
        self.transform = transform

    def __len__(self):
        return len(self.val_idxs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[self.val_idxs[idx], 0]))
        image = read_image(img_path + '.jpg')
        label = self.img_labels.iloc[self.val_idxs[idx], 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# This class of dataset is for the test data
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.input_paths = sorted(glob.glob(img_dir+'*'))
    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        img_path = self.input_paths[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path

# This class of dataset is for the semisupervised approach
class Middle_Dataset(Dataset):
    def __init__(self, transform=None, transform_2=None, transform_3=None, sel_test_data=None, sel_test_labels=None):
        self.transform = transform
        self.transform_2 = transform_2
        self.transform_3 = transform_3
        self.input_paths = sel_test_data
        self.label = sel_test_labels
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = self.input_paths[idx]
        image = read_image(img_path)
        label =  self.label[idx]

        if self.transform and self.transform_2 and self.transform_3:
            ra = random.random()
            # apply different transforms based on different probabilities
            if ra < 0.3:
                image = self.transform(image)
            elif ra < 0.6:
                image = self.transform_2(image)
            else:
                image = self.transform_3(image)

        elif self.transform and self.transform_2:
            if random.random() > 0.5:
                image = self.transform(image)
            else:
                image = self.transform(image)
        elif self.transform:
            image = self.transform(image)
        return image, label
        