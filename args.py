import argparse
import glob
import os
import time

global args
model_names = ['msdnet']

arg_parser = argparse.ArgumentParser(
                description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')
exp_group.add_argument('--resume', action='store_true',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--evalmode', default='dynamic',
                       choices=['anytime', 'dynamic'],
                       help='which mode to evaluate')
exp_group.add_argument('--evaluate-from', default='/home/liuliang/MSDNet_pre/models/step=4/msdnet-step=4-block=5.pth.tar', type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--gpu', default= '1' , type=str, help='GPU available.')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='ImageNet',
                        choices=['cifar10', 'cifar100', 'ImageNet'],
                        help='data to work on')
# data_group.add_argument('--data-root', metavar='DIR', default='/home/liuliang/dataset_python',
#                         help='path to dataset (default: data)')

data_group.add_argument('--data-root', metavar='DIR', default='/media/data/data02/Imagenet2012/',
                        help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true',
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

# model arch related
arch_group = arg_parser.add_argument_group('arch',
                                           'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='msdnet',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: msdnet)')
arch_group.add_argument('--reduction', default=0.5, type=float,
                        metavar='C', help='compression ratio of DenseNet'
                        ' (1 means dot\'t use compression) (default: 0.5)')

# msdnet config
arch_group.add_argument('--nBlocks', type=int, default=5)
arch_group.add_argument('--nChannels', type=int, default=32)
arch_group.add_argument('--base', type=int,default=4)
arch_group.add_argument('--stepmode', default='even', type=str, choices=['even', 'lin_grow'])
arch_group.add_argument('--step', type=int, default=4)
arch_group.add_argument('--growthRate', type=int, default=16)
arch_group.add_argument('--grFactor', default='1-2-4-4', type=str)
arch_group.add_argument('--prune', default='max', choices=['min', 'max'])
arch_group.add_argument('--bnFactor', default='1-2-4-4')
arch_group.add_argument('--bottleneck', default=True, type=bool)


# training related
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')

optim_group.add_argument('--epochs', default=300, type=int, metavar='N',
                         help='number of total epochs to run (default: 164)')
optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
optim_group.add_argument('-b', '--batch-size', default=256, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.1)')
optim_group.add_argument('--lr-type', default='multistep', type=str, metavar='T',
                        help='learning rate strategy (default: multistep)',
                        choices=['cosine', 'multistep'])
optim_group.add_argument('--decay-rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)')


