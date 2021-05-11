import os
import argparse
import json
import logging
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


is_download = False


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for test')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='momentum value for SGD')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='the number of steps until to log the training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training () tcp, gloo on cpu and gloo, nccl on gpu')

    # container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    # parser.add_argument('--hosts', type=list, default=[])
    # parser.add_argument('--current-host', type=str, default=None)
    # parser.add_argument('--model-dir', type=str, default='/tmp/mnist_model')
    # parser.add_argument('--data-dir', type=str, default='/Users/yameng/workspace/datasets/mnist_torch/train')
    # parser.add_argument('--num-gpus', type=int, default=0)

    return parser.parse_args()


def get_train_dataloader(batch_size, training_dir,
                         is_distributed=False, **kwargs):
    """
    create training data loader
    :param batch_size:
    :param training_dir:
    :param is_distributed:
    :param kwargs:
    :return:
    """
    logger.info("get train data loader")
    dataset = torchvision.datasets.MNIST(training_dir,
                                         train=True,
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,),
                                                                                            (0.3081,))]),
                                         download=is_download)

    train_sampler = torch.utils.data.DistributedSampler(dataset) if is_distributed else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, **kwargs)


def get_test_dataloader(batch_size, training_dir, **kwargs):
    """
    create test data loader
    :param batch_size:
    :param training_dir:
    :param kwargs:
    :return:
    """
    logger.info("get test data loader")
    return torch.utils.data.DataLoader(torchvision.datasets.MNIST(training_dir, train=False,
                                                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                                                transforms.Normalize(
                                                                                                    (0.1307,),
                                                                                                    (0.3081,))]),
                                                                  download=is_download),
                                       batch_size=batch_size, shuffle=True, **kwargs)


def average_gradients(model):

    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


def save_model(model, model_dir):
    logger.info("Saving the model")
    ckpt_path = os.path.join(model_dir, 'model.pth')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.cpu().state_dict(), ckpt_path)


class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=512, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv2(x)), 2))
        x = x.view(-1, 512)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    logger.info("Test set: Average loss: {:.4f}, "
                "Accuracy: {}/{} ({:.0f}%)\n".format(test_loss,
                                                     correct,
                                                     len(test_loader.dataset),
                                                     100. * correct / len(test_loader.dataset)))


def train(model, train_loader, optimizer, device, is_distributed, use_cuda, log_interval, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if is_distributed and not use_cuda:
            # average gradients manually for multi-machine cpu case only
            average_gradients(model)
        optimizer.step()
        if batch_idx % log_interval == 0:
            logger.info("Train epoch: {} [{}/{} ({:.0f})%] Loss: {:.6f}".format(epoch, batch_idx * len(data),
                                                                                len(train_loader.sampler),
                                                                                100. * batch_idx / len(train_loader),
                                                                                loss.item()))


def main():
    args = parse_arguments()

    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info("Initialized the distributed environment: \'{}\' backend on {} node. ".format(
            args.backend, dist.get_world_size() + 'Current host rank is {}. Number of gpus: {}'.format(dist.get_rank(),
                                                                                                       args.num_gpus)
        ))

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = get_train_dataloader(batch_size=args.batch_size,
                                        training_dir=args.data_dir,
                                        is_distributed=is_distributed,
                                        **kwargs)
    test_loader = get_test_dataloader(batch_size=args.batch_size,
                                      training_dir=args.data_dir,
                                      **kwargs)

    model = Net().to(device)
    # inputs = torch.zeros((4, 1, 28, 28))

    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, device, is_distributed, use_cuda, args.log_interval, epoch)
        test(model, test_loader, device)

    save_model(model, args.model_dir)


if __name__ == '__main__':
    main()
