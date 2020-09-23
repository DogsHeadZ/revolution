import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils
import models
from standard_dataloader import get_standard_loader
from meta_dataloader import get_meta_loader


def train(config):
    #### set the save and log path ####
    svname = args.name
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.set_save_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### make datasets ####
    # train
    dataloader_train = get_standard_loader(config['dataset_path'], config['train_dataset'], **config['train_dataset_args'])
    utils.log('train dataset: {}'.format(config['train_dataset']))
    # val
    if config.get('val_dataset'):
        eval_val = True
        dataloader_val = get_standard_loader(config['dataset_path'], config['val_dataset'], **config['val_dataset_args'])
    else:
        eval_val = False
    # few-shot test
    if config.get('test_dataset'):
        test_epoch = config.get('test_epoch')
        if test_epoch is None:
            ef_epoch = 5
        test_fs = True

        n_way = 5
        n_query = 15
        n_shots = [1, 5]
        dataloaders_test = []
        for n_shot in n_shots:
            dataloader_test = get_meta_loader(config['dataset_path'], config['test_dataset'], ways=n_way, shots=n_shot, query_shots=n_query, **config['test_dataset_args'])
            dataloaders_test.append(dataloader_test)
    else:
        test_fs = False

    #### Model and Optimizer ####
    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

    if test_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if test_fs:
            fs_model = nn.DataParallel(fs_model)

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    #### train and test ####

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.

    for epoch in range(1, max_epoch + 1):

        aves_keys = ['tl', 'ta', 'vl', 'va']
        if test_fs:
            for n_shot in n_shots:
                aves_keys += ['fsa-' + str(n_shot)]

        # train
        model.train()
        for data, label in tqdm(dataloader_train, desc='train', leave=False):
            data, label = data.cuda(), label.cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logits = None
            loss = None

        # eval
        if eval_val:
            model.eval()
            for data, label in tqdm(dataloader_val, desc='val', leave=False):
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    logits = model(data)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)

        if test_fs and (epoch % ef_epoch == 0 or epoch == max_epoch):
            fs_model.eval()
            for i, n_shot in enumerate(n_shots):
                np.random.seed(0)
                for data in tqdm(dataloaders_test[i],
                                    desc='fs-' + str(n_shot), leave=False):
                    train_inputs, train_targets = data["train"]
                    train_inputs = train_inputs.cuda()
                    train_targets = train_targets.cuda()
                    query_inputs, query_targets = data["test"]

                    with torch.no_grad():
                        logits = fs_model(train_inputs.view(train_inputs.shape[0], n_way, n_shot, *train_inputs.shape[-3:]), query_inputs).view(-1, n_way)
                        acc = utils.compute_acc(logits, label)
        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    train(config)
