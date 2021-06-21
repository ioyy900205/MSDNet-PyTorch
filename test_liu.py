'''
Date: 2021-05-22 12:11:47
LastEditors: Liuliang
LastEditTime: 2021-05-28 10:34:31
Description: 
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from args import arg_parser




def main(args):    
    # if args.gpu:
    #     print(args.gpu)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    c = torch.randn(7)

    c = c.cuda()

    print(c)
    




if __name__ == '__main__':    
    
    args = arg_parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    # print(torch.cuda.device_count())
    main(args)

