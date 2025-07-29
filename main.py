# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import datetime
import math
import os
import os.path as osp
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm
import collections
import warnings
from config import config
from data.data_manager import DataManager
from utils.avgmeter import AverageMeter
from utils.ci import mean_confidence_interval
from utils.iotools import save_checkpoint
from utils.logger import Logger
from utils.losses import CrossEntropyLoss, DistillKL, JS_Divergence
from utils.optimizers import init_optimizer
from utils.torchtools import adjust_learning_rate, one_hot
from models.BSEM import BSEM  # Binocular Sensing Extractor Module(BSEM)
from models.IAPM import IAPM  # Instance Authentication Perception Module(IAPM)
from joblib import Parallel, delayed
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch.distributed as dist
try: from apex import amp
except Exception:
    amp = None
    print('WARNING: could not import pygcransac')
    pass
tqdm_kwargs = {'bar_format': '{l_bar}{bar:30}{r_bar}', 'colour': '#32CD32'}
warnings.filterwarnings('ignore')

def main(args):
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else: print("Currently using CPU (GPU is highly recommended)")

    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()
    
    model = BSEM(args, scale_cls=args.scale_cls, num_classes=args.num_classes)
    
    criterion = CrossEntropyLoss()
    assert args.disparity in ['kl', 'js']
    if args.disparity =='kl': criterion_disparity = DistillKL(T=args.T)
    elif args.disparity =='js': criterion_disparity = JS_Divergence(T=args.T)

    optimizer = init_optimizer(args.optim, model, args.lr, args.weight_decay)
    if args.amp_opt_level != "O0" and amp is not None:  model, optimizer = amp.initialize(model, optimizer, opt_level=config.amp_opt_level)
    
    best_acc = -np.inf
    best_epoch = 0

    checkpoint_flag = args.checkpoint
    if (args.resume is None) and checkpoint_flag=='True':
        model = model.to('cuda')
        log_dir = osp.join(args.save_dir, 'checkpoint_ep'+'60.pth.tar')
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print('加载checkpoint处模型成功,将从Epoch{}继续训练!'.format(args.start_epoch+1))
    elif (args.resume is None): print('不加载预训练模型，从头开始训练! ')
    
    if args.resume is not None:
        state_dict = torch.load(args.resume)['state_dict']
        model.load_state_dict(state_dict)
        print('Load model from {}'.format(args.resume))
        
    if use_gpu:
        model = model.cuda()
        if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
    
    if args.mode == 'test':
        meta_test(model, testloader, use_gpu, args)
        return

    # 5.meta_train和meta_val=================================================================
    start_time = time.time()
    train_time = 0
    learning_rate = args.lr
    print("==> Start training")
    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        # meta_train
        meta_train(epoch, model, criterion, criterion_disparity, optimizer, trainloader, learning_rate, use_gpu, args)

        train_time += round(time.time() - start_train_time)
        # meta-val
        if epoch == 0 or epoch > (args.stepsize[0]-1) or (epoch + 1) % 10 == 0:
            acc = meta_val(epoch, model, testloader, use_gpu, args)
            # save
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
            if isinstance(model, torch.nn.DataParallel): state_dict = model.module.state_dict()
            else: state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'acc': acc,
                'best_acc': best_acc,
                'optimizer':optimizer.state_dict(),
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))
        
        # update lr
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)

    # record time
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))

# meta-training
def meta_train(epoch, model, criterion, criterion_disparity, optimizer, trainloader, learning_rate, use_gpu, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    tqdm_gen = tqdm.tqdm(trainloader, **tqdm_kwargs)

    model.train()
    end = time.time()
    for i, batch in enumerate(tqdm_gen, 1):    
        images_train, labels_train, images_test, labels_test, pids = batch[0], batch[1], batch[2], batch[3], batch[4]
        data_time.update(time.time() - end)
        
        if use_gpu:
            images_train, labels_train = images_train.cuda(), labels_train.cuda()
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids = pids.cuda()
        labels_train_1hot = one_hot(labels_train).cuda()
        labels_test_1hot  = one_hot(labels_test).cuda()

        lcls_scores, lcls_scores2, lfeat_scores, labels_test1, rcls_scores, rcls_scores2, rfeat_scores, labels_test2 = model(images_train, images_test, labels_train_1hot, labels_test_1hot, labels_train, labels_test, epoch, args.max_epoch)
        lcls_scores=lcls_scores.view(lcls_scores.shape[0], lcls_scores.shape[1], -1)
        lcls_scores2=lcls_scores2.view(lcls_scores2.shape[0], lcls_scores2.shape[1], -1)
        rcls_scores=rcls_scores.view(rcls_scores.shape[0], rcls_scores.shape[1], -1)
        rcls_scores2=rcls_scores2.view(rcls_scores2.shape[0], rcls_scores2.shape[1], -1)

        # =================== cal loss =====================
        lsfe_loss = 0
        for i in range(lcls_scores.size(2)): lsfe_loss += criterion(lcls_scores[..., i], pids) / lcls_scores.size(2)
        lsfe_loss += criterion(lcls_scores2, labels_test1)
        rsfe_loss = 0
        for i in range(rcls_scores2.size(2)): rsfe_loss += criterion(rcls_scores2[..., i], pids) / rcls_scores2.size(2)
        rsfe_loss += criterion(rcls_scores, labels_test2)
        if args.disparity =='js': mutual_loss = criterion_disparity(lfeat_scores, rfeat_scores) + criterion_disparity(rfeat_scores, lfeat_scores)
        loss = args.w[0] * lsfe_loss + args.w[1] * rsfe_loss + args.w[2] * mutual_loss

        tqdm_gen.set_description('Epoch {}/{},Train Loss={:.4f}'.format(epoch + 1, args.max_epoch, loss))

        # backwards
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            if amp is not None:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: # not support amp
                loss.backward()
        else:
            loss.backward()
        optimizer.step()        
        torch.cuda.synchronize()

        losses.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(epoch+1, learning_rate, batch_time=batch_time, data_time=data_time, loss=losses))

# meta-validation
def meta_val(epoch, model, testloader, use_gpu, args):
    accs = AverageMeter()
    test_accuracies = []
    tqdm_gen = tqdm.tqdm(testloader, **tqdm_kwargs)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm_gen, 1):    
            images_train, labels_train, images_test, labels_test, _ = batch[0], batch[1], batch[2], batch[3], batch[4]

            batch_size = images_train.size(0)        
            num_test_examples = images_test.size(1)  
            labels_train_1hot = one_hot(labels_train)
            labels_test_1hot  = one_hot(labels_test) 
            
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()
                labels_train_1hot = labels_train_1hot.cuda()
                labels_test_1hot = labels_test_1hot.cuda()

            cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot, labels_train, labels_test, epoch, args.max_epoch)
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            _, preds = torch.max(cls_scores.detach().cpu(), 1)

            labels_test = labels_test.view(batch_size * num_test_examples)
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)  # acc: tensor(0.3867)
            accs.update(acc.item(), labels_test.size(0))
            
            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy()

            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)
            tqdm_gen.set_description('Val Batch {}'.format(i))

    accuracy = accs.avg  
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    mean_acc, ci = mean_confidence_interval(test_accuracies) 
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(mean_acc, ci))

    return accuracy

# meta-testing
def meta_test(model, testloader, use_gpu, args):
    iapm = IAPM(args, classifier=args.classifier, num_class=args.nKnovel, ratio=args.expand_ratio, lreduce=args.lembed, rreduce=args.rembed, d=args.emb_dim, logit_penalty=args.logit_penalty)
    acc_list = [[] for _ in range(2)]
    tqdm_gen = tqdm.tqdm(testloader, **tqdm_kwargs)

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for inter, batch in enumerate(tqdm_gen, 1):    
            images_train, labels_train, images_test, labels_test, images_unlabel = batch[0], batch[1], batch[2], batch[3], batch[4]

            assert images_train.shape[0] == 1  
            num_train = images_train.shape[1] 
            num_test = images_test.shape[1]    

            labels_train = labels_train.squeeze(0).numpy().reshape(-1)  
            labels_test  = labels_test.squeeze(0).numpy().reshape(-1)           

            if args.unlabel != 0: images = torch.cat([images_train, images_test, images_unlabel],1).squeeze(0) 
            else: images = torch.cat([images_train, images_test],1).squeeze(0)
            if use_gpu: 
                images = images.cuda()                          
                images_test = images_test.cuda()
            
            lembeddings = model.branch1.left_extractor_reduction(images).detach().cpu().numpy()   
            ltrain_embeddings = lembeddings[:num_train]                         
            ltest_embeddings = lembeddings[num_train:num_train+num_test]                   
            if args.unlabel != 0: lunlabel_embeddings = lembeddings[num_train+num_test:]
            else: lunlabel_embeddings = None

            rembeddings = model.branch2.right_extractor_reduction(images).detach().cpu().numpy()  
            rtrain_embeddings = rembeddings[:num_train]                         
            rtest_embeddings = rembeddings[num_train:num_train+num_test]                   
            if args.unlabel != 0: runlabel_embeddings = rembeddings[num_train+num_test:]
            else: runlabel_embeddings = None
            

            iapm.nerve_conduct(ltrain_embeddings, rtrain_embeddings, labels_train)
            acc = iapm.perception(ltest_embeddings, rtest_embeddings, args, lunlabel_embeddings, runlabel_embeddings, True, labels_test)  # show_detail=True
            
            for i in range(len(acc)): acc_list[i].append(acc[i])
            acc_list[-1].append(acc[-1])
            print("Test Epoch {}/{}, Predict_I Acc={:.4f}, Predict_II Acc={:.4f}".format(inter, args.epoch_size, acc[0], acc[1]))    

    end_time = time.time()
    run_time = end_time - start_time
    minutes = int(run_time / 60)  
    seconds = int(run_time % 60)  
    print("程序运行时间：{} 分钟 {} 秒".format(minutes, seconds))

    mean_acc_list = []
    ci_list = []

    for i, item in enumerate(acc_list):
        mean_acc, ci = mean_confidence_interval(item)
        mean_acc_list.append(mean_acc)
        ci_list.append(ci)
    print('--------------------------')
    print(" "*7 + "Pred_I " + "Pred_II")
    print("MEAN:  {}".format('   '.join([str(i*100)[:5] for i in mean_acc_list])))      
    print("95%CI: {}".format('  '.join([str(i*100)[:5] for i in ci_list])))             
    print('--------------------------')
    return

if __name__ == '__main__':
    args = config()
    cudnn.benchmark = True
    main(args)
