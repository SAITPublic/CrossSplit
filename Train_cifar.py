
from __future__ import print_function
from ipaddress import v6_int_to_packed
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
from PreResNet_cifar import *
import dataloader_cifar as dataloader
from math import log2
from Contrastive_loss import *
import torch.distributed as dist
import collections.abc
from collections.abc import MutableMapping
from numpy import array, exp
import time

def func(x, a, b, c):
    return a*exp(-b*x)+c 

# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

## Arguments to pass 
parser = argparse.ArgumentParser(description='PyTorch CrossSplit CIFAR Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--metric', type=str, default = 'JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='../datasets/cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', default='PRN18', type=str)
parser.add_argument('--delt', default=10, type=int, help='relaxation')
args = parser.parse_args()

## INFO
fname  = './ckpt_CIFAR_'+ args.arch + '/'
print('name: ', fname)
print('lr: ', args.lr)

if args.dataset=='cifar10':
    warm_up = 10 
elif args.dataset=='cifar100':
    warm_up = 30

## GPU Setup 
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

## Download the Datasets
if args.dataset== 'cifar10':
    args.data_path = '../datasets/cifar-10'
    torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)
else:
    args.data_path = '../datasets/cifar-100'
    args.num_class = 100
    torchvision.datasets.CIFAR100(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path,train=False, download=True)

## Checkpoint Location
folder = args.dataset + '_' + args.noise_mode + '_' + str(args.r)+'_' + str(args.lr)+'_' + str(args.batch_size)+'_d' + str(args.delt)
print(folder)

model_save_loc = fname + folder
if not os.path.exists(fname):
    os.mkdir(fname)
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

## Log files
stats_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     
test_loss_log = open(model_save_loc +'/test_loss.txt','w')

## For Standard Training 
def warmup_standard(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
      
        optimizer.zero_grad()
        _, outputs = net(inputs)   

        loss_CE = CEloss(outputs, labels) 

        if args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise           
            penalty = conf_penalty(outputs)

            L = loss_CE + 0.1*penalty 
        else:   
            L = loss_CE
        #L = loss_CE

        L.backward()  
        optimizer.step()                
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_CE.item()))
        sys.stdout.flush()

def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, Jmax, Jmin):
    
    JS_dist = Jensen_Shannon()
    net.train()
    net2.eval()

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1

    ## Loss statistics
    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4, _, _ = unlabeled_train_iter.next() #_, _
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4, _, _ = unlabeled_train_iter.next() #_, _
        
        batch_size = inputs_x.size(0)
        label_x0 = labels_x

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
        
        with torch.no_grad():

            ## Unlabeled dataset: Label co-guessing of unlabeled samples using self prediction by net 
            _, outputs_u11 = net(inputs_u)
            _, outputs_u12 = net(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2

            ptu = pu**(1/args.T) ## Temparature Sharpening   
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()                  

            ## Cross-split label correction 
            ## Labeled dataset: Label refinement of labeled samples using peer prediction by net2 
            _, outputs_x_peer  = net2(inputs_x)
            _, outputs_x2_peer = net2(inputs_x2)        

            px_peer = (torch.softmax(outputs_x_peer, dim=1) + torch.softmax(outputs_x2_peer, dim=1)) / 2
           
            w_x0 = JS_dist(labels_x,  px_peer)
            
            # Equation (4): Normalize the JSD through shifting and scaling
            for b in range(0, batch_size): 
                w_x[b] = (w_x0[b] - Jmin[label_x0[b]])/(Jmax[label_x0[b]] - Jmin[label_x0[b]])
            
            w_x[w_x>1] = 1.0
            w_x[w_x<0] = 0.0
           
            if epoch < warm_up + args.delt*2: 
                gamma = 0.6
            elif epoch < warm_up + args.delt*3: 
                gamma = 0.8
            else: 
                gamma = 1.0

            # Equation (2): Set beta (=w_x)   
            w_x = (w_x - 0.5)*gamma+ 0.5            
            w_x = w_x.view(-1).cuda()

            # Equation (1): s_i = beta*y_peer + (1-beta)*y_label
            px = w_x.unsqueeze(1) * px_peer + (1-w_x.unsqueeze(1)) * labels_x      

            ptx = px**(1/args.T)    ## Temparature sharpening                    
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
            targets_x = targets_x.detach()

        ## Unsupervised Contrastive Loss
        f1, _ = net(inputs_u3)
        f2, _ = net(inputs_u4)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_simCLR = contrastive_criterion(features)

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
        all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b   = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        ## Mixup
        mixed_input  = l * input_a  + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        _, logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]
        
        ## Combined Loss
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        ## Regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        ## Total Loss
        loss = Lx + lamb * Lu + args.lambda_c*loss_simCLR + penalty

        ## Accumulate Loss
        loss_x += Lx.item()
        loss_u += Lu.item()
        # loss_ucl += loss_simCLR.item()

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f Contrastive Loss:%.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss_x/(batch_idx+1), loss_u/(batch_idx+1),  loss_ucl/(batch_idx+1)))
        sys.stdout.flush()

## Test Accuracy
def test_ensemble(epoch,net1,net2):
    net1.eval()
    net2.eval()

    num_samples = 1000
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1 = net1(inputs)  
            _, outputs2 = net2(inputs)           
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)            
            loss = CEloss(outputs, targets)  
            loss_x += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()  

    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write(str(acc)+'\n')
    test_log.flush()  
    test_loss_log.write(str(loss_x/(batch_idx+1))+'\n')
    test_loss_log.flush()
    return acc


# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)


## Calculate JSD's Min and Max
def Calculate_MinMax_JSD(model1, num_samples):  
    JS_dist = Jensen_Shannon()
    JSD   = torch.zeros(num_samples)    

    class_list = [None]*args.num_class
    Jmax = [None]*args.num_class
    Jmin = [None]*args.num_class
    for i in range(0, args.num_class):
        class_list[i] = []

    for batch_idx, (inputs, targets, _, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])  
           
        ## Get the Prediction
        out = out1 

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
        JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist
        
        for b in range(0, batch_size):
            class_list[targets[b]].append(dist[b].cpu()) 
    
    for i in range(0, args.num_class):
        Jmax[i] = np.max(class_list[i])
        Jmin[i] = np.min(class_list[i])

    return JSD, Jmax, Jmin

## Unsupervised Loss coefficient adjustment 
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
       
        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    if args.arch == 'PRN18':
        model = ResNet18(num_classes=args.num_class)
    elif args.arch == 'PRN34': 
        model = ResNet34(num_classes=args.num_class)
        
    model = model.cuda()
    return model

## Choose Warmup period based on Dataset
num_samples = 50000

## Call the dataloader
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
    root_dir=model_save_loc,log=stats_log, noise_file='%s/clean_%.4f_%s.npz'%(args.data_path,args.r, args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
net1 = nn.DataParallel(net1)
net2 = nn.DataParallel(net2)
cudnn.benchmark = True

## Semi-Supervised Loss
criterion  = SemiLoss()

## Optimizer and Scheduler
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 300, 2e-3)

optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 300, 2e-3)

## Loss Functions
CE       = nn.CrossEntropyLoss(reduction='none')
CEloss   = nn.CrossEntropyLoss()
MSE_loss = nn.MSELoss(reduction= 'none')
contrastive_criterion = SupConLoss()

if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

## Resume from the warmup checkpoint 
model_name_1 = 'Net1_warmup.pth'  
model_name_2 = 'Net2_warmup.pth'  
if args.resume:
    start_epoch = warm_up
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1))['net'])
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2))['net'])
else:
    start_epoch = 0

best_acc = 0
all_loss = [[],[]] # save the history of losses from two networks

## Warmup and SSL-Training     

test_loader = loader.run('test')
warmup_trainloader = loader.run('warmup')
train_loaderset1, train_loaderset2, train_loaderset1s, train_loaderset2s = loader.run('train_split') #random split
eval_loader = loader.run('eval')
for epoch in range(start_epoch,args.num_epochs+1):   


    ## Warmup Stage 
    if epoch<warm_up:       
        print('Warmup Training')
        warmup_standard(epoch, net1, optimizer1, warmup_trainloader) 
        warmup_standard(epoch, net2, optimizer2, warmup_trainloader) 
    
    else:
        
        start_time = time.time()
        
        print('\nTraining - 1st\n')
        ## Calculate JSD's Min and Max values - 1st
        prob, Jmax, Jmin = Calculate_MinMax_JSD(net2, num_samples)           
        train(epoch,net1,net2,optimizer1,train_loaderset1, train_loaderset2s, Jmax, Jmin) #labeled: trainingset1, unlabeled: trainingset2

        print('\nTraining - 2nd\n')       
        ## Calculate JSD's Min and Max values - 2nd
        prob, Jmax, Jmin = Calculate_MinMax_JSD(net1, num_samples)        
        train(epoch,net2,net1,optimizer2,train_loaderset2, train_loaderset1s, Jmax, Jmin) #labeled: trainingset2, unlabeled: trainingset1
        
        print("\n--{}s seconds--".format(time.time()-start_time))
        
    acc = test_ensemble(epoch,net1,net2)

    scheduler1.step()
    scheduler2.step()

    if acc > best_acc:
        if epoch <warm_up:
            model_name_1 = 'Net1_warmup.pth'
            model_name_2 = 'Net2_warmup.pth'
        else:
            model_name_1 = 'Net1.pth'  
            model_name_2 = 'Net2.pth'  

        print("Save the Model-----")
        checkpoint1 = {
            'net': net1.state_dict(),
            'Model_number': 1,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Dataset': 'Cifar',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }
        checkpoint2 = {
            'net': net2.state_dict(),
            'Model_number': 2,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Dataset': 'Cifar',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
        best_acc = acc

