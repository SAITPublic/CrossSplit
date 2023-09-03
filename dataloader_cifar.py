from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
from autoaugment import CIFARPolicy, ImageNetPolicy
import torch.nn.functional as F 
from Asymmetric_Noise import *

from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log='', didx=[]): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode
        root_dir_save = root_dir

        if dataset == 'cifar10':
            root_dir = '../datasets/cifar-10/'            
            num_class =10         
        else:
            root_dir = '../datasets/cifar-100/'
            num_class =100

        ## For Asymmetric Noise (CIFAR10)    
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} 

        num_sample     = 50000
        self.class_ind = {}

        if self.mode=='test':
            if dataset=='cifar10':    
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            
            if os.path.exists(noise_file):             
                noise_label = np.load(noise_file)['label']
                noise_idx = np.load(noise_file)['index']
                idx       = list(range(50000))
                clean_idx = [x for x in idx if x not in noise_idx]
                for kk in range(num_class):
                    self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]

            else:       ## Inject Noise   
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)            
                noise_idx = idx[:num_noise]
                
                if noise_mode == 'asym':
                    if dataset== 'cifar100':
                        noise_label, prob11 =  noisify_cifar100_asymmetric(train_label, self.r)
                    else:
                        for i in range(50000):
                            if i in noise_idx:
                                    noiselabel = self.transition[train_label[i]]
                                    noise_label.append(noiselabel)
                            else:
                                noise_label.append(train_label[i])   
                else:
                    for i in range(50000):
                        if i in noise_idx:
                            if noise_mode=='sym':
                                if dataset=='cifar10': 
                                    noiselabel = random.randint(0,9)
                                elif dataset=='cifar100':    
                                    noiselabel = random.randint(0,99)
                                noise_label.append(noiselabel)

                            elif noise_mode=='pair_flip':  
                                noiselabel = self.pair_flipping[train_label[i]]
                                noise_label.append(noiselabel)   
                    
                        else:
                            noise_label.append(train_label[i])   

                print("Save noisy labels to %s ..."%noise_file)        
                np.savez(noise_file, label = noise_label, index = noise_idx)          
                for kk in range(num_class):
                    self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]    

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label  

            elif self.mode == 'all_train':
                self.train_data = train_data
                self.noise_label = noise_label
                self.train_label = train_label   
                
            elif self.mode == 'all_eval':
                self.train_data = train_data
                self.noise_label = noise_label
                self.train_label = train_label               
                                       

    def __getitem__(self, index):
        if self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index   
            
        elif self.mode=='all_eval':
            img, target, target_gt = self.train_data[index], self.noise_label[index], self.train_label[index]
            image = Image.fromarray(img)
            img = self.transform(image)
          
            return img, target, target_gt, index 

        elif self.mode=='all_train':
            img, target, prob = self.train_data[index], self.noise_label[index], 1.0
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)    
            return img1, img2, img3, img4,  target, prob 

        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)   
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        
        if self.dataset=='cifar10':
            transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_weak_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_strong_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFARPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_10,
                "unlabeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
                "labeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
                "all_train": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
            }

            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])

        elif self.dataset=='cifar100':
            transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            transform_weak_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            transform_strong_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFARPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_100,
                "unlabeled": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
                "labeled": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
                "all_train": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
            }        
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
                   
    def run(self, mode, pred=[], prob=[], didx=0):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["warmup"], mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader

        elif mode=='eval':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all_eval', noise_file=self.noise_file)      
            
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers, drop_last= True)          
            return eval_loader          

        elif mode=='train_split':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["all_train"], mode="all_train",noise_file=self.noise_file)  

            random_seed= 42
            dataset_size = len(all_dataset)

            split = int(np.floor(0.5 * dataset_size))
            
            train_dataset1, train_dataset2 = torch.utils.data.random_split(all_dataset, [split, split])
        
            train_loader1 = DataLoader(train_dataset1.dataset,batch_size=self.batch_size,shuffle=True)
            train_loader1s = DataLoader(train_dataset1.dataset,batch_size=int(self.batch_size*0.5),shuffle=True)
            train_loader2 = DataLoader(train_dataset2.dataset,batch_size=self.batch_size,shuffle=True)
            train_loader2s = DataLoader(train_dataset2.dataset, batch_size=int(self.batch_size*0.5),shuffle=True)
            return train_loader1, train_loader2, train_loader1s, train_loader2s
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
