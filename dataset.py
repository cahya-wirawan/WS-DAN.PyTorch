""" NOTICE: A Custom Dataset SHOULD BE PROVIDED
Created: May 02,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd

__all__ = ['CustomDataset']


class CustomDataset(Dataset):
    """
    # Description:
        Basic class for retrieving images and labels

    # Member Functions:
        __init__(self, phase, shape):   initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            shape:                      output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, datapath, phase='train', shape=(512, 512)):
        assert phase in ['train', 'val', 'test']
        self.datapath = datapath
        self.phase = phase
        self.files = (datapath/phase).glob("**/*.jpg")
        carnames = datapath/"names.csv"
        df = pd.read_csv(carnames, header=None)
        car_labels = {name: i for i,name in enumerate(list(df[0]))}
        self.data_list = []

        for fullpath in self.files:
            filepath = str(fullpath).split('/')
            self.data_list.append((fullpath, car_labels[filepath[-2]]))

        self.shape = shape

        # transform
        self.transform = transforms.Compose([
            transforms.Resize(size=(self.shape[0], self.shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        image = Image.open(self.data_list[item][0]).convert('RGB')  # (C, H, W)
        image = self.transform(image)
        assert image.size(1) == self.shape[0] and image.size(2) == self.shape[1]

        if self.phase != 'test':
            # filename of image should have 'id_label.jpg/png' form
            label = self.data_list[item][1]  # label
            return image, label
        else:
            # filename of image should have 'id.jpg/png' form, and simply return filename in case of 'test'
            return image, str(self.data_list[item][0])

    def __len__(self):
        return len(self.data_list)
