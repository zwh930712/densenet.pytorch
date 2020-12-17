#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math

import shutil

import setproctitle
import simplenet
import make_graph
from dataset.datasets import ClsDataset
import dataset.transforms as transform
from utils.timer import Timer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=16)
    parser.add_argument('--train-data', type=str, default="/home/wenhai.zhang/WORK_SPACE/cdeg/DataSet/ESOPHAGUS_20201207/txt/train.csv")
    parser.add_argument('--test-data', type=str, default="/home/wenhai.zhang/WORK_SPACE/cdeg/DataSet/ESOPHAGUS_20201207/txt/test.csv")
    parser.add_argument('--nEpochs', type=int, default=1000)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--save', type=str, default="/home/wenhai.zhang/cls_net")
    parser.add_argument('--crop_size', type=list, default=[64, 64])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    trainTransform = transform.Compose([
        transform.CenterCrop(args.crop_size),
        transform.Resize(args.crop_size),
        transform.NormLize(),
        transforms.ToTensor(),
    ])

    testTransform = transform.Compose([
        transform.CenterCrop(args.crop_size),
        transform.Resize(args.crop_size),
        transform.NormLize(),
        transforms.ToTensor(),
    ])

    train_dataset = ClsDataset(args.train_data, imgsz=(512, 512), crop_size=args.crop_size, transforms=trainTransform, p_ratio=0.5)
    test_dataset = ClsDataset(args.test_data, imgsz=(512, 512), crop_size=args.crop_size, transforms=testTransform, p_ratio=0.5)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchSz,
        shuffle=True,
        drop_last=True,
        **kwargs)

    testLoader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batchSz,
        shuffle=False,
        drop_last=True,
        **kwargs)

    net = simplenet.SimpleNet(3, 2)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    BestTestErr = 1.0

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test_err = test(args, epoch, net, testLoader, optimizer, testF)
        print("epoch:{}, test err:{}".format(epoch, test_err))
        if test_err < BestTestErr:
            BestTestErr = test_err
            torch.save(net, os.path.join(args.save, 'best.pth'))
        # os.system('./plot.py {} &'.format(args.save))
        torch.save(net, os.path.join(args.save, 'latest.pth'))
    trainF.close()
    testF.close()


def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 1.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}  \tLR: {:.4f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader), loss.data, err, optimizer.param_groups[0]["lr"]))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data, err))
        trainF.flush()


def test(args, epoch, net, testLoader, optimizer, testF):
    timer = Timer()
    timer.tic()
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()
    average_time = timer.toc()
    print("test a img average time is {:.3f} seconds".format(average_time / len(testLoader.dataset)))
    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 1.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f})\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

    return err


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-2
        elif epoch == 150: lr = 1e-3
        elif epoch == 225: lr = 1e-4
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__=='__main__':
    main()
