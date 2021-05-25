'''
Date: 2021-05-22 12:11:47
LastEditors: Liuliang
LastEditTime: 2021-05-24 17:39:37
Description: 
'''

import os
import torch
from args import arg_parser
import models
from net_measure import measure_model
from model import AlexNet

def main(args):    
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    
    def get_num_gen(gen):
        return sum(1 for x in gen)


    def is_leaf(model):
        return get_num_gen(model.children()) == 0


    model_1 = getattr(models,'msdnet')
    # print(model_1)
    # model_2 = models.msdnet(args)
    # print(model_2)

    model_2 = AlexNet()

    a = model_2.children()    
    a_1 = next(a)    
    a_2 = next(a)
    
    b = a_1.children()
    b_1 = next(b)
    b_2 = next(b)
    b_3 = next(b)
    b_4 = next(b)

    # def a(m):
    #     def b(x):
    #         print('a')
    #         return 5
    #     return b

    # c = a(1)
    # print(c)
    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']
    print(scores)

    
    


    n_flops, n_params = measure_model(model_2, IM_SIZE, IM_SIZE) 
    print(n_flops,n_params)
    



if __name__ == '__main__':    
    args = arg_parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.nScales = len(args.grFactor)

    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']

    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 1000

    torch.manual_seed(args.seed)
   
    
    # 准确率

    

    # 创建文件夹
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
        # IM_SIZE = 32
        IM_SIZE = 224
    else:
        IM_SIZE = 224
    main(args)

