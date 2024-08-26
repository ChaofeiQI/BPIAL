# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import math,time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.vision_res12_le import LE_vision_res12
from models.vision_res12_re import RE_vision_res12

class LSFE(nn.Module):
    def __init__(self, scale_cls, iter_num_prob=35.0/75, num_classes=64, method=1, FFT_sign=False, Elastic=False):
        super(LSFE, self).__init__()
        self.scale_cls = scale_cls
        self.iter_num_prob = iter_num_prob 
        self.method = method
        self.FFT_sign = FFT_sign
        self.Elastic = Elastic
        self.lbase = LE_vision_res12(method) 
        self.lnFeat = self.lbase.nFeat   
        self.lclasifier = nn.Conv2d(self.lnFeat, num_classes, kernel_size=1) 

    def left_extractor_reduction(self, x): 
        f = self.lbase(x, self.method, self.FFT_sign)    
        f = f.mean(2).mean(2)                            
        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12)
        return f

    def extractor_test(self, ftrain, ftest):  
        ftrain, ftest = torch.relu(ftrain), torch.relu(ftest)
        ftrain = ftrain.mean(3).mean(3)  
        ftest = ftest.mean(3).mean(3)    
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12) 
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)    
        scores = self.scale_cls * torch.bmm(ftest, ftrain.transpose(1,2))
        return scores

    def compute_logits(self, proto, query, emb_dim, query_idx):
        num_batch = proto.shape[0]  
        num_proto = proto.shape[1]  
        num_query = query_idx.shape[-1]  
        query = query.reshape(-1, emb_dim).unsqueeze(1)  
        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)  
        proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)  
        logits = torch.sum((proto - query) ** 2, 2)  
        return logits

    def image_elastic_constraint(self, n_ways, logits, label, epoch, max_epoch):
        ep_weight = epoch / max_epoch                 
        constraint = torch.zeros_like(logits).cuda()  
        for idx in range(logits.shape[0]):  
            cur_dis = logits[idx].detach().cpu()  
            cur_label = label[idx].detach().cpu() 
            pos_dis = cur_dis[cur_label]          
            mask = (torch.arange(0, n_ways) == cur_label).int()  
            neg_hard_dis, _ = torch.topk(cur_dis * (1 - mask), k=2, largest=False)  
            assert neg_hard_dis[0] == 0 and _[0] == cur_label
            prob = torch.softmax(-cur_dis, dim=0)            
            prob_pos, prob_neg = prob[_[0]], prob[_[1]]
            
            if prob_neg > 0.5 or prob_pos > 0.5:  
                constraint[idx] = 0
            else:
                beta = torch.sigmoid(0.1 * (pos_dis - neg_hard_dis[1]))
                constraint[idx] = mask * (ep_weight + 1e-8) * beta * 5.5
        return constraint

    def le_sensing(self, xtrain, xtest, ytrain, ytest, labels_train, labels_test, n_ways, epoch=None, max_epoch=None):
        batch_size, num_train = xtrain.size(0), xtrain.size(1) 
        num_test = xtest.size(1)                                
        K = ytrain.size(2)                                      
        ytrain = ytrain.transpose(1, 2) 
        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4)) 
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))      
        x = torch.cat((xtrain, xtest), 0) 
        
        f = self.lbase(x, self.method, self.FFT_sign)
        h, w = f.shape[2], f.shape[3]
        ftrain = f[:batch_size * num_train]              
        ftrain = ftrain.view(batch_size, num_train, -1)  
        ftrain = torch.bmm(ytrain, ftrain)  
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain)) 
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:]) 
        
        ftest = f[batch_size * num_train:]               
        ftest = ftest.view(batch_size, num_test, *f.size()[1:]) 
        ftest_call = ftest.flatten(start_dim=3).mean(dim=-1)    

        if not self.training: 
            support, query = ftrain.flatten(start_dim=2), ftest.flatten(start_dim=2)
            support = support.view(support.shape[0],support.shape[1]//n_ways,n_ways,support.shape[2])
            proto = support.mean(dim=1) 
            emb = support.size(-1)
            query_idx = labels_test
            logits = - self.compute_logits(proto, query, emb, query_idx)   
            return logits

        logits = torch.zeros((batch_size*num_test, K)).cuda()          
        query = torch.mean(ftest.view(ftest.shape[0],ftest.shape[1],ftest.shape[2],-1),dim=-1)   
        query_idx = labels_test  
        label = labels_test.view(-1) 
        support = ftrain.view(ftrain.shape[0],ftrain.shape[1]//n_ways,n_ways,ftrain.shape[2],-1) 
        proto = support.mean(dim=1).mean(dim=-1)  
        emb = proto.size(-1) 
        
        cur_logits = self.compute_logits(proto, query, emb, query_idx)  

        if self.Elastic=='1':
            constraint = self.image_elastic_constraint(K, cur_logits, label, epoch, max_epoch) 
            cur_logits = - (cur_logits + constraint)
            logits = cur_logits           
        else: 
            logits = -cur_logits          )

        elastic_scores=logits.view(logits.shape[0],logits.shape[1]) 
    
        b, n1, c, h, w = ftrain.size()  
        n2 = ftest.size(1)  
        ftest  = ftest.view(b, n2, c, -1)                
        ftest  = ftest.unsqueeze(1).repeat(1,1,n1,1,1)   
        ftest  = ftest.view(b, n1, n2, c, h, w).transpose(1, 2) 
        ftest_cls = ftest.view(batch_size, num_test, K, -1) 
        ftest_cls = ftest_cls.transpose(2, 3)     
        ytest = ytest.unsqueeze(3)         
        ftest_cls = torch.matmul(ftest_cls, ytest) 
        ftest_cls = ftest_cls.view(batch_size * num_test, *f.size()[1:])
        ytest = self.lclasifier(ftest_cls) 

        ftrain = ftrain.view(b, n1, c, -1)                      
        ftrain = ftrain.unsqueeze(2).repeat(1,1,n2,1,1)         
        ftrain = ftrain.view(b, n1, n2, c, h, w).transpose(1, 2)
        ftrain = ftrain.mean(4).mean(4)                         
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4) 
        ftrain_norm = ftrain_norm.unsqueeze(5) 
        ftest_norm  = F.normalize(ftest, p=2, dim=3, eps=1e-12) 
        cls_scores  = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)   
        cls_scores  = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])
        return ytest, elastic_scores, ftest_call.view(-1,ftest_call.shape[2])       

    def forward(self, xtrain, xtest, ytrain, ytest, labels_train, labels_test, trainable, n_ways, epoch=None, max_epoch=None):
        if not trainable: 
            lcls_scores = self.le_sensing(xtrain, xtest, ytrain, ytest, labels_train, labels_test, n_ways, epoch, max_epoch)
            return lcls_scores
        else:
            lcls_scores, elastic_scores, lfeat_scores = self.le_sensing(xtrain, xtest, ytrain, ytest, labels_train, labels_test, n_ways, epoch, max_epoch)
            return lcls_scores, elastic_scores, lfeat_scores, labels_test

class RSFE(nn.Module):
    def __init__(self, scale_cls, iter_num_prob=35.0/75, num_classes=64, method=1, FFT_sign=False, Elastic=False):
        super(RSFE, self).__init__()
        self.scale_cls = scale_cls
        self.iter_num_prob = iter_num_prob 
        self.method = method
        self.FFT_sign = FFT_sign
        self.Elastic = Elastic
        self.rbase = RE_vision_res12(method)    
        self.rnFeat = self.rbase.nFeat   
        self.rclasifier = nn.Conv2d(self.rnFeat, num_classes, kernel_size=1) 

    def right_extractor_reduction(self, x): 
        f = self.rbase(x, self.method, self.FFT_sign)     
        f = f.mean(2).mean(2)                             
        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12) 
        return f

    def extractor_test(self, ftrain, ftest):  
        ftrain, ftest = torch.relu(ftrain), torch.relu(ftest)
        ftrain = ftrain.mean(3).mean(3) 
        ftest = ftest.mean(3).mean(3)   
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)    
        scores = self.scale_cls * torch.bmm(ftest, ftrain.transpose(1,2))
        return scores

    def compute_logits(self, proto, query, emb_dim, query_idx):
        num_batch = proto.shape[0]  
        num_proto = proto.shape[1]  
        num_query = query_idx.shape[-1] 
        query = query.reshape(-1, emb_dim).unsqueeze(1)  
        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)  
        proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)  
        logits = torch.sum((proto - query) ** 2, 2) 
        return logits

    def image_elastic_constraint(self, n_ways, logits, label, epoch, max_epoch):
        ep_weight = epoch / max_epoch                
        constraint = torch.zeros_like(logits).cuda() 
        for idx in range(logits.shape[0]):  
            cur_dis = logits[idx].detach().cpu()  
            cur_label = label[idx].detach().cpu() 
            pos_dis = cur_dis[cur_label]          
            mask = (torch.arange(0, n_ways) == cur_label).int()  
            neg_hard_dis, _ = torch.topk(cur_dis * (1 - mask), k=2, largest=False) 
            assert neg_hard_dis[0] == 0 and _[0] == cur_label
            prob = torch.softmax(-cur_dis, dim=0)           
            prob_pos, prob_neg = prob[_[0]], prob[_[1]]
            if prob_neg > 0.5 or prob_pos > 0.5:  
                constraint[idx] = 0
            else:
                beta = torch.sigmoid(0.1 * (pos_dis - neg_hard_dis[1]))
                constraint[idx] = mask * (ep_weight + 1e-8) * beta * 5.5
        return constraint
    
    def re_sensing(self, xtrain, xtest, ytrain, ytest, labels_train, labels_test, n_ways, epoch=None, max_epoch=None):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)  
        num_test = xtest.size(1)                               
        K = ytrain.size(2)                                     
        ytrain = ytrain.transpose(1, 2)
        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))     
        x = torch.cat((xtrain, xtest), 0)  
        
        f = self.rbase(x, self.method, self.FFT_sign)
        h, w = f.shape[2], f.shape[3]

        ftrain = f[:batch_size * num_train]              
        ftrain = ftrain.view(batch_size, num_train, -1)  
        ftrain = torch.bmm(ytrain, ftrain)  
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))  
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:]) 
        ftest = f[batch_size * num_train:]               
        ftest = ftest.view(batch_size, num_test, *f.size()[1:]) 
        ftest_call = ftest.flatten(start_dim=3).mean(dim=-1)    

        if not self.training: 
            support, query = ftrain.flatten(start_dim=2), ftest.flatten(start_dim=2)
            support = support.view(support.shape[0],support.shape[1]//n_ways,n_ways,support.shape[2])
            proto = support.mean(dim=1) 
            emb = support.size(-1) 
            query_idx = labels_test
            logits = - self.compute_logits(proto, query, emb, query_idx)  
            return logits

        logits = torch.zeros((batch_size*num_test, K)).cuda()         
        query = torch.mean(ftest.view(ftest.shape[0],ftest.shape[1],ftest.shape[2],-1),dim=-1)  
        query_idx = labels_test  
        label = labels_test.view(-1) 
        support = ftrain.view(ftrain.shape[0],ftrain.shape[1]//n_ways,n_ways,ftrain.shape[2],-1)
        proto = support.mean(dim=1).mean(dim=-1) 
        emb = proto.size(-1)
        
        cur_logits = self.compute_logits(proto, query, emb, query_idx)  

        if self.Elastic=='1':
            constraint = self.image_elastic_constraint(K, cur_logits, label, epoch, max_epoch) 
            cur_logits = - (cur_logits + constraint)
            logits = cur_logits            
        else: 
            logits = -cur_logits          
        elastic_scores=logits.view(logits.shape[0],logits.shape[1])
        
        b, n1, c, h, w = ftrain.size()  
        n2 = ftest.size(1) 
        ftest  = ftest.view(b, n2, c, -1)                
        ftest  = ftest.unsqueeze(1).repeat(1,1,n1,1,1)   
        ftest  = ftest.view(b, n1, n2, c, h, w).transpose(1, 2) 
        ftest_cls = ftest.view(batch_size, num_test, K, -1)
        ftest_cls = ftest_cls.transpose(2, 3)      
        ytest = ytest.unsqueeze(3)         
        ftest_cls = torch.matmul(ftest_cls, ytest)
        ftest_cls = ftest_cls.view(batch_size * num_test, *f.size()[1:])
        ytest = self.rclasifier(ftest_cls) 
        
        ftrain = ftrain.view(b, n1, c, -1)                      
        ftrain = ftrain.unsqueeze(2).repeat(1,1,n2,1,1)         
        ftrain = ftrain.view(b, n1, n2, c, h, w).transpose(1, 2)
        ftrain = ftrain.mean(4).mean(4)                         
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12) 
        ftrain_norm = ftrain_norm.unsqueeze(4) 
        ftrain_norm = ftrain_norm.unsqueeze(5) 
        ftest_norm  = F.normalize(ftest, p=2, dim=3, eps=1e-12) 
        cls_scores  = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)   
        cls_scores  = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:]) 
        return elastic_scores, ytest, ftest_call.view(-1,ftest_call.shape[2]) 

    def forward(self, xtrain, xtest, ytrain, ytest, labels_train, labels_test, trainable, n_ways, epoch=None, max_epoch=None):
        if not trainable: 
            logits = self.re_sensing(xtrain, xtest, ytrain, ytest, labels_train, labels_test, n_ways, epoch, max_epoch)     
            return logits 
        else:
            relastic_scores, ytest, rfeat_scores = self.re_sensing(xtrain, xtest, ytrain, ytest, labels_train, labels_test, n_ways, epoch, max_epoch)
            return relastic_scores, ytest, rfeat_scores, labels_test

class BSEM(nn.Module):
    def __init__(self, args, scale_cls, iter_num_prob=35.0/75, num_classes=64):
        super(BSEM, self).__init__()
        self.n_train_ways = args.n_train_ways
        self.nKnovel = args.nKnovel
        self.method = args.method
        self.FFT_sign = args.FFT_sign
        self.Elastic = args.Elastic
        self.branch1 = LSFE(scale_cls, iter_num_prob, num_classes, self.method, self.FFT_sign, self.Elastic)
        self.branch2 = RSFE(scale_cls, iter_num_prob, num_classes, self.method, self.FFT_sign, self.Elastic)

    def forward(self, xtrain, xtest, ytrain, ytest, labels_train, labels_test, epoch=None, max_epoch=None):
        if not self.training:
            lcls_logits = self.branch1.forward(xtrain, xtest, ytrain, ytest, labels_train, labels_test, self.training, self.nKnovel, epoch, max_epoch)
            rcls_logits = self.branch2.forward(xtrain, xtest, ytrain, ytest, labels_train, labels_test, self.training, self.nKnovel, epoch, max_epoch) 
            return lcls_logits + rcls_logits 
        else:
            lcls_score, lcls_score2, lfeat_scores, labels_test1 = self.branch1.forward(xtrain, xtest, ytrain, ytest, labels_train, labels_test, self.training, self.n_train_ways, epoch, max_epoch)
            rcls_score, rcls_score2, rfeat_scores, labels_test2 = self.branch2.forward(xtrain, xtest, ytrain, ytest, labels_train, labels_test, self.training, self.n_train_ways, epoch, max_epoch)
            return lcls_score, lcls_score2, lfeat_scores, labels_test1, rcls_score, rcls_score2, rfeat_scores, labels_test2   
