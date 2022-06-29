# encoding: utf-8
"""
Read images and corresponding labels.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import copy

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class BaseDataset(Dataset):
    def __init__(self, root_dir, images, labels, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(BaseDataset, self).__init__()
        self.root_dir = root_dir
        self.images = copy.deepcopy(images)
        self.labels = copy.deepcopy(labels)
        self.transform = transform
        self.image2label = {}
        self.last_images = copy.deepcopy(images)
        self.last_labels = copy.deepcopy(labels)
        for i, image_name in enumerate(self.images):
            self.image2label[image_name] = copy.deepcopy(self.labels[i])
        # print('Total # samples :{}'.format(len(self.images)))
    
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        items = self.images[index]
        if '.png' not in self.images[index]:
            image_name = os.path.join(self.root_dir, self.images[index] + '.jpg')
        else:
            image_name = os.path.join(self.root_dir, self.images[index])
        
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return items, index, image, label
    
    def re_load(self):
        self.images = copy.deepcopy(self.last_images)
        self.labels = copy.deepcopy(self.last_labels)
    
    def update(self, images):
        self.images = copy.deepcopy(images)
        for i, image_name in enumerate(self.images):
            self.labels[i] = copy.deepcopy(self.image2label[image_name])
             

    def __len__(self):
        return len(self.images)

class CSVDataset(BaseDataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        
        self.file = pd.read_csv(csv_file)
        
        self.root_dir = root_dir
        self.images = self.file['ImageID'].values
        self.labels = self.file.iloc[:, 1:].values.astype(int)
        
        self.transform = transform


class DatasetSplit(Dataset):
     def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        # print('Total # samples:{}'.format(len(self.idxs)))

     def __len__(self):
        return len(self.idxs)

     def __getitem__(self, item):
        items, index, image,  label = self.dataset[self.idxs[item]]
        return items, index, image, label

def onehot_reverse(labels):
    assert len(labels.shape)==2
    new_labels = [np.argmax(l) for l in labels]

    return new_labels



def get_Pi(sets, classnum=5, noniid=False):
    Pi = []

    for i in range(sets):
        # randomly bank prior
        this_Pi = np.random.rand(classnum) * 0.9 + 0.1
        this_Pi[i % classnum] *= 2
        # normalize to sum=1
        this_Pi = this_Pi / np.sum(this_Pi)
        Pi.append(this_Pi)

    Pi = np.array(Pi)
    return Pi

def get_sub_bank_sizes(sub_banks, data_len):

    sub_bank_size = data_len // sub_banks
    sub_bank_sizes = np.ones(sub_banks) * sub_bank_size
    return sub_bank_sizes



def get_sub_banks(sub_bank_num_perclient, y_train, y_indices, sub_bank_sizes, thetas, classnum=5):
    class_idx = []

    for cls in range(classnum):
        this_idx = [y_indices[i] for i, x in enumerate(y_train) if x == cls]
        class_idx.append(this_idx)

    sub_banks = ()
    size_bag = []
    start_idx = [0 for i in range(classnum)]
    for i in range(sub_bank_num_perclient):
        size_cls = []
        if i < sub_bank_num_perclient-1:
            for cls in range(classnum):
                n_this = int(sub_bank_sizes[i] * thetas[i][cls])    
                if cls == 0:
                    cur_sub_bank = np.array(class_idx[cls][start_idx[cls]:start_idx[cls]+n_this])
                else:
                    cur_sub_bank = np.concatenate((cur_sub_bank, class_idx[cls][start_idx[cls]:start_idx[cls]+n_this])).astype(int)

                size_cls.append(len(class_idx[cls][start_idx[cls]:start_idx[cls]+n_this]))
                start_idx[cls] += len(class_idx[cls][start_idx[cls]:start_idx[cls]+n_this])
                np.random.shuffle(cur_sub_bank)

        # fill the remaining samples
        else:
            for cls in range(classnum):

                if cls == 0:
                    cur_sub_bank = np.array(class_idx[cls][start_idx[cls]:])
                else:
                    cur_sub_bank = np.concatenate((cur_sub_bank, class_idx[cls][start_idx[cls]:])).astype(int)

                size_cls.append(len(class_idx[cls][start_idx[cls]:]))
                start_idx[cls] += len(class_idx[cls][start_idx[cls]:])
                np.random.shuffle(cur_sub_bank)
        # concatenate different class data
        sub_banks = sub_banks + (torch.from_numpy(cur_sub_bank),)
        size_bag.append(np.array(size_cls) / sum(size_cls))

    # calculate priors corr for every sub-bank
    sub_banks_num_count = [len(sub_banks[j]) for j in range(len(sub_banks))]
    priors_corr = torch.from_numpy(
        np.array([sub_banks_num_count[k] / sum(sub_banks_num_count) for k in range(len(sub_banks_num_count))]))

    bags_pi = np.array(size_bag)

    return sub_banks, priors_corr, bags_pi


def get_class_index(targets, classnum=5):
    indexs = []

    for cls in range(classnum):
        this_index = [index for (index, value) in enumerate(targets) if value == cls]
        indexs.append(this_index)

    return indexs


def load_data(args, trian_data, dict_users, clientnum=10, sub_bank_num_perclient=5, classnum=5, noniid=True):
    all_train_data = trian_data
    client_train_data = []
    client_Pi = []
    client_priors_corr = []

    # for every client
    for n in range(clientnum):
        this_Pi = torch.from_numpy(get_Pi(sub_bank_num_perclient, classnum=classnum))
        this_sub_bank_sizes = get_sub_bank_sizes(sub_bank_num_perclient, len(dict_users[n]))
        this_sub_banks, this_priors_corr, this_Pi = get_sub_banks(sub_bank_num_perclient,
                                                                       onehot_reverse(all_train_data.labels[list(dict_users[n])]),
                                                                       list(dict_users[n]),
                                                                       this_sub_bank_sizes, this_Pi, classnum=classnum)
        client_Pi.append(torch.from_numpy(this_Pi))
        client_priors_corr.append(this_priors_corr)
        client_bank_temp_data = None
        client_bank_temp_targets = None
        for i in range(sub_bank_num_perclient):
            this_sub_bank_temp_data = list(all_train_data.images[this_sub_banks[i].numpy()])
            #  sub-bank index as proxy labels
            this_sub_bank_temp_targets = torch.ones(len(this_sub_banks[i])) * i
            # concatenate data and labels
            if i == 0:
                client_bank_temp_data = this_sub_bank_temp_data
                client_bank_temp_targets = this_sub_bank_temp_targets
            else:
                client_bank_temp_data.extend(this_sub_bank_temp_data)
                client_bank_temp_targets = torch.cat((client_bank_temp_targets, this_sub_bank_temp_targets))
          
        client_train_data.append({'images': client_bank_temp_data, 'labels': client_bank_temp_targets})
       
    return client_train_data, client_priors_corr, client_Pi







