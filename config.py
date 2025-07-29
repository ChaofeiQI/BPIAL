# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import argparse

def config():
    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='miniImageNet')
    parser.add_argument('--load', default=False)
    parser.add_argument('-j', '--workers', default=8, type=int, help="number of data loading workers (default: 4)")  # 11-2
    parser.add_argument('--height', type=int, default=84, help="height of an image (default: 84)")
    parser.add_argument('--width', type=int, default=84, help="width of an image (default: 84)")

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--scale_cls', type=int, default=7)
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('-g', '--gpu-devices', default='0', type=str)
    parser.add_argument('--checkpoint', default=False)

    # ************************************************************
    # BPIAL parameters
    # ************************************************************
    parser.add_argument('--classifier', type=str, default='lr')
    parser.add_argument('--disparity', type=str, default='js')
    parser.add_argument("--T", type=float, default=4.0)
    parser.add_argument('--lembed', type=str, default='se')
    parser.add_argument('--rembed', type=str, default='se')
    parser.add_argument('--emb_dim', type=int, default=5)
    parser.add_argument('--logit_penalty', type=float, default=0.5)
    parser.add_argument('--confidence_ratio', type=float, default=0.6)
    parser.add_argument("--method", type=str, choices=['1','2','3'], default='2')  # Input method
    parser.add_argument("--FFT_sign", default=False)  # Input method
    parser.add_argument("--Elastic", type=str, default='1', choices=['0','1'], help='elastic_constraint') # [0(Flase),1(True)]

    parser.add_argument("--weights", type=str, default="1-1-1") 
    parser.add_argument("--fit_mode", type=str, default="db", choices=['db'])
    parser.add_argument('--expand_ratio', type=float, default=0.5)
    parser.add_argument("--stereopsis", type=str, default="intersection", choices=['intersection','union','le','re'])
    parser.add_argument("--expand_style", type=str, default="linear", choices=['linear'])
    
    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--phase', default='test', type=str, help='use test or val dataset to early stop')
    
    parser.add_argument('--n_train_ways', type=int, default=15, metavar='N', help='Number of classes for doing each classification run')        
    parser.add_argument('--train_nTestNovel', type=int, default=7 * 15, help='number of test examples for all the novel category when training') 

    parser.add_argument('--nKnovel', type=int, default=5, help='number of novel categories')                                                    
    parser.add_argument('--meta_nTestNovel', type=int, default=7 * 5, help='number of test examples for all the novel category when training') 
    parser.add_argument('--nExemplars', type=int, default=5, help='number of training examples per novel category.')                            
    parser.add_argument('--unlabel', type=int, default=0)                                                                                       
    parser.add_argument('--nTestNovel', type=int, default=15 * 5, help='number of test examples for all the novel category')                    

    parser.add_argument('--start-epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument('--max-epoch', default=90, type=int, help="maximum epochs to run")
    parser.add_argument('--stepsize', default=[60], nargs='+', type=int, help="stepsize to decay learning rate") 

    parser.add_argument('--train_epoch_size', type=int, default=1200, help='number of batches per epoch when training')
    parser.add_argument('--epoch_size', type=int, default=2000, help='number of batches per epoch')

    parser.add_argument('--test-batch', default=1, type=int, help="test batch size")
    parser.add_argument('--train-batch', default=2, type=int, help="train batch size")  # 11-11

    # ************************************************************
    # Optimization options
    # ************************************************************    
    parser.add_argument('--optim', type=str, default='SGD', help="optimization algorithm (see optimizers.py)")   # sgd,SGD
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,  help="initial learning rate")       # 初始学习率
    parser.add_argument("--decay_step", type=int, default=40)
    parser.add_argument('--LUT_lr', default=[(60, 0.1), (80, 0.006), (90, 0.0012)], help="multistep to decay learning rate")  # optim='sgd'时使用
    parser.add_argument('--weight-decay', default=5e-04, type=float,  help="weight decay (default: 5e-04)")

    # ************************************************************
    # DistributedDataParallel
    # ************************************************************
    parser.add_argument('--amp_opt_level', type=str, default='O0', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument("--seed", default='1', type=str)
    
    # ***********************************************************
    args = parser.parse_args()

    args.w = [float(i) for i in args.weights.split('-')]
    args.train_epoch_size= int(600*args.train_batch)

    if args.dataset=='CIFARFS':
        args.num_classes = 64
        args.dataset_dir = '/home/ssdData/qcfData/fs_benchmarks/CIFARFS/'
    elif args.dataset=='CUB_Croped':
        args.num_classes = 100
        args.dataset_dir = '/home/ssdData/qcfData/fs_benchmarks/CUB_fewshot_cropped'
    elif args.dataset=='Aircraft':
        args.num_classes = 50
        args.dataset_dir = '/home/ssdData/qcfData/fs_benchmarks/Aircraft_fewshot/'
    elif args.dataset=='TieredImagenet':
        args.num_classes = 351
        args.train_epoch_size = 13980
        args.max_epoch = 120
        args.LUT_lr = [(30, 0.1), (60, 0.01), (90, 0.001),(120, 0.0001)]
        args.dataset_dir = '/home/ssdData/qcfData/fs_benchmarks/TieredImagenet_224'
    elif args.dataset=='MiniImagenet':
        args.num_classes = 64
        args.dataset_dir = '/home/ssdData/qcfData/fs_benchmarks/mini_imagenet_full_size/mini_imagenet_full_size'
    else:
        print("No dataset supported.")

    return args
