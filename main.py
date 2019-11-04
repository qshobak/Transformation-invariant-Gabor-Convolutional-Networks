from __future__ import division
import os
import time
import argparse
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from utils import accuracy, AverageMeter, save_checkpoint, visualize_graph, get_parameters_size
from tensorboardX import SummaryWriter
from net_factory import get_network_fn

import numpy as np
import random
# RandomRotate
import numbers
import math
from PIL import Image, ImageOps

parser = argparse.ArgumentParser(description='PyTorch GCN MNIST Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--gpu', default=-1, type=int,
                    metavar='N', help='GPU device ID (default: -1)')
parser.add_argument('--dataset_dir', default='../../MNIST', type=str, metavar='PATH',
                    help='path to dataset (default: ../MNIST)')
parser.add_argument('--comment', default='', type=str, metavar='INFO',
                    help='Extra description for tensorboard')
parser.add_argument('--model', default='', type=str, metavar='NETWORK',
                    help='Network to train')
args = parser.parse_args()


use_cuda = (args.gpu >= 0) and torch.cuda.is_available()
best_prec1 = 0
writer = SummaryWriter(comment='_'+args.model+'_'+args.comment)
iteration = 0

# custom transform
class RandomRotate(object):
    """Rotate the given PIL.Image counter clockwise around its centre by a random degree 
    (drawn uniformly) within angle_range. angle_range is a tuple (angle_min, angle_max). 
    Empty region will be padded with color specified in fill."""
    def __init__(self, angle_range=(-180,180), fill='black'):
        assert isinstance(angle_range, tuple) and len(angle_range) == 2 and angle_range[0] <= angle_range[1]
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.angle_range = angle_range
        self.fill = fill

    def __call__(self, img):
        angle_min, angle_max = self.angle_range
        angle = angle_min + random.random() * (angle_max - angle_min)
        theta = math.radians(angle)
        w, h = img.size
        diameter = math.sqrt(w * w + h * h)
        theta_0 = math.atan(float(h) / w)
        w_new = diameter * max(abs(math.cos(theta-theta_0)), abs(math.cos(theta+theta_0)))
        h_new = diameter * max(abs(math.sin(theta-theta_0)), abs(math.sin(theta+theta_0)))
        pad = math.ceil(max(w_new - w, h_new - h) / 2)
        img = ImageOps.expand(img, border=int(pad), fill=self.fill)
        img = img.rotate(angle, resample=Image.BICUBIC)
        return img.crop((pad, pad, w + pad, h + pad))

normalize = transforms.Normalize((0.1307,), (0.3081,))
train_transform = transforms.Compose([
    transforms.Resize(32),
    RandomRotate((-180, 180)),
    transforms.ToTensor(),
    normalize,
    ])
test_transform = transforms.Compose([
    transforms.Resize(32),
    RandomRotate((-180, 180)),
    transforms.ToTensor(), 
    normalize,
    ])


train_dataset = datasets.MNIST(root=args.dataset_dir, train=True, 
                    download=False, transform=train_transform)
test_dataset = datasets.MNIST(root=args.dataset_dir, train=False, 
                    download=False,transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True, shuffle=True)

# Load model
model = get_network_fn(args.model)
print(model)

# Try to visulize the model
try:
	visualize_graph(model, writer, input_size=(2, 1, 32, 32))
except:
	print('\nNetwork Visualization Failed! But the training procedure continue.')

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=3e-05)#code
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)#code--10, 0.5
#scheduler = StepLR(optimizer, step_size=25, gamma=0.5)#paper
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)
criterion = criterion.to(device)

# Calculate the total parameters of the model
print('Model size: {:0.2f} million float parameters'.format(get_parameters_size(model)/1e6))

if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))

def train(epoch):
    model.train()
    global iteration
    st = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        iteration += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)

        prec1, = accuracy(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), prec1.item()))
            writer.add_scalar('Loss/Train', loss.item(), iteration)
            writer.add_scalar('Accuracy/Train', prec1, iteration)
    epoch_time = time.time() - st
    lr = optimizer.param_groups[0]['lr']
    print('Epoch time:{:0.2f}s'.format(epoch_time),  '	learning-rate:', lr)
    scheduler.step()

def test(epoch):
    model.eval()
    test_loss = AverageMeter()
    acc = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss.update(F.cross_entropy(output, target, reduction='mean').item(), target.size(0))
            prec1, = accuracy(output, target) # test precison in one batch
            acc.update(prec1.item(), target.size(0))
    print('\nVal set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss.avg, acc.avg))
    writer.add_scalar('Loss/Test', test_loss.avg, epoch)
    writer.add_scalar('Accuracy/Test', acc.avg, epoch)
    return acc.avg

for epoch in range(args.start_epoch, args.epochs):
    print('------------------------------------------------------------------------')
    train(epoch+1)
    prec1 = test(epoch+1)

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)

print('Finished!')
print('Best Test Precision@top1:{:.2f}'.format(best_prec1))
writer.add_scalar('Best TOP1', best_prec1, 0)
writer.close()
