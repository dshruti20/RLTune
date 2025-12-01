from __future__ import print_function

import argparse
import os
import time
from subprocess import PIPE

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models # this is for importing popular models/architectures from Pytorch

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

def train(args, model, device, train_loader, criterion, optimizer, epoch, writer):
    model.train()    #set model in training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('loss', loss.item(), niter)

def test(args, model, device, test_loader, criterion, writer, epoch):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    writer.add_scalar('accuracy', top1.avg, epoch)

# ref: https://github.com/pytorch/examples/blob/2639cf050493df9d3cbf065d45e6025733add0f4/imagenet/main.py
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # possible popular models/architectures names provided by Pytorch
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training')
    # added arguments for architecture + dataset (+ downloaded imagenet_dataroot if dataset is imagenet).
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('--imagenet_dataroot', type=str, default='../imagenet_data', help='path to imagenet dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='supported datasets are: [mnist - cifar10 - cifar100 - imagenet].')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')
    parser.add_argument('--mode', type=str, default=None, choices=['dp', 'ddp'],
                    help='Choose between DataParallel (dp) and DistributedDataParallel (ddp)')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to use')
    #if dist.is_available():
        #parser.add_argument('--backend', type=str, help='Distributed backend',
                            #choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            #default=dist.Backend.GLOO)
    args = parser.parse_args()
    # DP: backend is gloo; DDP: backend is nccl
    #if args.mode == 'ddp':
       # backend = 'nccl'

    backend = 'nccl' if args.mode == 'ddp' else 'gloo'

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA')
        os.system('nvidia-smi --query-gpu=gpu_name,utilization.gpu,utilization.memory --format=csv')

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("\nDevice: %s\n" %device)

    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(backend))
        #dist.init_process_group(backend=args.backend)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # dataset: add multiple dataset options for the data
    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../MNIST_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(224),
                           transforms.Grayscale(3),  # Add this line to convert to 3 channels
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../MNIST_data', train=False, download=True,
                        transform=transforms.Compose([
                           transforms.Resize(224),
                           transforms.Grayscale(3),  # Add this line to convert to 3 channels
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../cifar10_data', train=True, download=True,
                            transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../cifar10_data', train=False,  download=True,
                        transform=transforms.Compose([
                           transforms.Resize(224),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../cifar100_data', train=True, download=True,
                            transform=transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../cifar100_data', train=False,  download=True,
                        transform=transforms.Compose([
                           transforms.Resize(224),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'imagenet':
        # to use imagenet the data should be already downloaded and processed
        # please specify --imagenet_dataroot which is the folder where train/val folders are
        # default --imagenet_dataroot is ../imagenet_data
        traindir = os.path.join(args.imagenet_dataroot, 'train')
        valdir = os.path.join(args.imagenet_dataroot, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    else:
        print ('Dataset not supported.\nSupported datasets are: [mnist - cifar10 - cifar100 - imagenet].')

    # model = Net().to(device)
    # set the model to the one specified by the arg --arch
    num_gpus = args.num_gpus
    if num_gpus == 1:
        print("Running on a single GPU, no parallelism.")
        model = models.__dict__[args.arch]().to(device)

    elif num_gpus > 1:
        if args.mode is None:
            raise ValueError("For multiple GPUs, you must specify --mode dp or ddp.")
        model = models.__dict__[args.arch]()

        if args.mode == 'dp':
            print(f"Using DataParallel on {num_gpus} GPUs.")
            model = model.to(device)
            if use_cuda:
                model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

        elif args.mode == 'ddp':
            print(f"Using DistributedDataParallel on {num_gpus} GPUs.")
            dist.init_process_group(backend=backend, init_method='env://')

            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)

            model = model.to(torch.device('cuda', local_rank))
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        raise ValueError("Invalid GPU configuration.")

    #if is_distributed():
        #Distributor = nn.parallel.DistributedDataParallel if use_cuda \
            #else nn.parallel.DistributedDataParallelCPU
        #model = Distributor(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    time_list = []
    test_time_list = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(args, model, device, train_loader, criterion, optimizer, epoch, writer)
        total_epoch_time = time.time() - start
        time_list.append(total_epoch_time)
        if use_cuda:
            os.system('nvidia-smi --query-gpu=gpu_name,utilization.gpu,utilization.memory --format=csv')
        start_test = time.time()
        test(args, model, device, test_loader, criterion, writer, epoch)
        total_epoch_test_time = time.time() - start_test
        test_time_list.append(total_epoch_test_time)

    total_time = sum(time_list)
    total_test_time = sum(test_time_list)
    print("Total time in seconds: %d" %total_time)
    print("Total time in seconds: %d" %total_test_time)

    if (args.save_model):
        torch.save(model.state_dict(),"%s_%s.pt"%(args.dataset, args.arch))

if __name__ == '__main__':
    main()
