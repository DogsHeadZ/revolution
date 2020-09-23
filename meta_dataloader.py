from torchmeta.datasets.helpers import (omniglot, miniimagenet, tieredimagenet,
                                        cub, cifar_fs, doublemnist, triplemnist)
from torchmeta.utils.data import BatchMetaDataLoader, NonEpisodicWrapper
import torch
from torch.utils.data import Dataset, DataLoader

def get_meta_loader(datapath, dataset, ways, shots, query_shots, batch_size,
                       split, download=True, shuffle=True, num_workers=0):
    """Create an episode data loader for a torchmeta dataset. Can also
    include unlabelled data for semi-supervised learning.

    dataset: String. Name of the dataset to use.
    datapath: String. Path, where dataset are stored.
    ways: Integer. Number of ways N.
    shots: Integer. Number of shots K for support set.
    test_shots: Integer. Number of images in query set.
    batch_size: Integer. Number of tasks per iteration.
    split: String. One of ['train', 'val', 'test']
    download: Boolean. Whether to download the data.
    shuffle: Boolean. Whether to shuffle episodes.
    """
    # Select dataset
    if dataset == 'omniglot':
        dataset_func = omniglot
    elif dataset == 'miniimagenet':
        dataset_func = miniimagenet
    elif dataset == 'tieredimagenet':
        dataset_func = tieredimagenet
    elif dataset == 'cub':
        dataset_func = cub
    elif dataset == 'cifar_fs':
        dataset_func = cifar_fs
    elif dataset == 'doublemnist':
        dataset_func = doublemnist
    elif dataset == 'triplemnist':
        dataset_func = triplemnist
    else:
        raise ValueError("No such dataset available. Please choose from\
                         ['omniglot', 'miniimagenet', 'tieredimagenet',\
                          'cub, cifar_fs, doublemnist, triplemnist']")

    # Collect arguments that are the same for all possible sub-datasets
    kwargs = {'download': download,
              'meta_train': split == 'train',
              'meta_val': split == 'val',
              'meta_test': split == 'test',
              'shuffle': shuffle}

    # Create dataset for labelled images
    dataset = dataset_func(datapath,
                           ways=ways,
                           shots=shots,
                           test_shots=query_shots,
                           **kwargs)

    return BatchMetaDataLoader(dataset, batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=torch.cuda.is_available())
