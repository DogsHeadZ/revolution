import os
import io
import h5py
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Standard_dataset(Dataset):
    def __init__(self, datapath, dataset, split, transform):
        self.data = self._extract_data_from_hdf5(dataset, datapath, split)
        self.transform = transform
        self.dataset= dataset

    def _extract_data_from_hdf5(self, dataset, datapath, split):
        datapath = os.path.join(datapath, dataset)
        # Load omniglot
        if dataset == 'omniglot':
            classes = []
            with h5py.File(os.path.join(datapath, 'data.hdf5'), 'r') as f_data:
                with open(os.path.join(datapath,
                          'vinyals_{}_labels.json'.format(split))) as f_labels:
                    labels = json.load(f_labels)
                    for label in labels:
                        img_set, alphabet, character = label
                        classes.append(f_data[img_set][alphabet][character][()])

        # Load mini-imageNet or cub
        elif dataset == 'miniimagenet' or dataset == 'cub':
            with h5py.File(os.path.join(datapath, split + '_data.hdf5'), 'r') as f:
                datasets = f['datasets']
                print(datasets.keys())
                data = np.concatenate([datasets[k] for k in datasets.keys()], axis=0)
                self.label = [i // 600 for i in range(data.shape[0])]
        else:
            raise ValueError("No such dataset available. Please choose from\
                             ['omniglot', 'miniimagenet']")

        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.dataset == 'cub':
            image = Image.open(io.BytesIO(self.data[index])).convert('RGB')
        else:
            image = Image.fromarray(self.data[index])

        return self.transform(image), self.label[index]


def get_standard_loader(datapath, dataset, batch_size, split='train', shuffle=True, num_workers=0, **kwargs):
    img_size = (28, 28) if dataset == 'omniglot' else (84, 84)
    # Get transform
    augment = kwargs.get('augment')
    if augment == 'aug1':
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                              saturation=0.4, hue=0.1)
        transform = transforms.Compose([transforms.RandomResizedCrop(size=img_size[-2:],
                                                                       scale=(0.5, 1.0)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize(img_size),
                               transforms.ToTensor()])

    dataset = Standard_dataset(dataset, datapath, split, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return dataloader

if __name__ == '__main__':
    dataset_train = Standard_dataset('miniimagenet', 'few_data/', 'train')