# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import os
import sys
import time
import scipy
import math
import copy
import numpy as np
import glmnet_python              
from sklearn.linear_model import ElasticNet 
from sklearn.preprocessing import normalize 
from joblib import Parallel, delayed

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class IAPM(object):
    def __init__(self, args, classifier='lr', num_class=None, ratio=0.6, max_iter='auto', lreduce='lle', rreduce='lle', d=5, norm='l2', logit_penalty=0.5):
        self.step = math.ceil(10*ratio) 
        self.max_iter = max_iter     
        self.num_class = num_class    
        self.confidence_ratio= args.confidence_ratio
        self.step_max=args.nTestNovel//5
        self.initial_norm(norm)       
        self.initial_embed(lreduce,rreduce,d)
        self.initial_classifier(classifier)  
        self.logit_penalty = logit_penalty   
    
    def label2onehot(self, label, num_class):
        result = np.zeros((label.shape[0], num_class))
        for ind, num in enumerate(label): result[ind, num] = 1.0
        return result
    
    def initial_norm(self, norm):
        norm = norm.lower()
        assert norm in ['l2', 'none']
        if norm == 'l2': self.norm = lambda x: normalize(x)
        else: self.norm = lambda x: x
    
    def initial_embed(self, lreduce, rreduce, d):
        from sklearn.manifold import MDS
        from sklearn.manifold import Isomap
        from sklearn.decomposition import PCA
        from sklearn.manifold import LocallyLinearEmbedding
        from sklearn.manifold import SpectralEmbedding
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        lreduce, rreduce = lreduce.lower(), rreduce.lower()
        assert lreduce in ['mds', 'pca', 'isomap', 'ltsa', 'lle', 'se', 'lda', 'none']
        assert rreduce in ['mds', 'pca', 'isomap', 'ltsa', 'lle', 'se', 'lda', 'none']
        
        if lreduce == 'lle':  lembed = LocallyLinearEmbedding(n_components=d, n_neighbors=5,eigen_solver='dense') 
        elif lreduce == 'se': lembed = SpectralEmbedding(n_components=d) 
        elif lreduce == 'lda': lembed = LinearDiscriminantAnalysis(n_components=d) 
        elif lreduce == 'mds': lembed = MDS(n_components=d, metric=False)
        elif lreduce == 'pca': lembed = PCA(n_components=d)
        elif lreduce == 'isomap': lembed = Isomap(n_components=d)
        elif lreduce == 'ltsa': lembed = LocallyLinearEmbedding(n_components=d, n_neighbors=5, method='ltsa')
        
        if rreduce == 'lle':  rembed = LocallyLinearEmbedding(n_components=d, n_neighbors=5,eigen_solver='dense') 
        elif rreduce == 'se': rembed = SpectralEmbedding(n_components=d) 
        elif rreduce == 'lda': rembed = LinearDiscriminantAnalysis(n_components=d) 
        elif rreduce == 'mds': rembed = MDS(n_components=d, metric=False)
        elif rreduce == 'pca': rembed = PCA(n_components=d)
        elif rreduce == 'isomap': rembed = Isomap(n_components=d)
        elif rreduce == 'ltsa': rembed = LocallyLinearEmbedding(n_components=d, n_neighbors=5, method='ltsa')
        
        if lreduce == 'none': self.lembed = lambda x: x
        else: self.lembed = lambda x: lembed.fit_transform(x)
        if rreduce == 'none': self.rembed = lambda x: x
        else: self.rembed = lambda x: rembed.fit_transform(x)
        

    def initial_classifier(self, classifier):
        assert classifier in ['svm', 'knn', 'lr', 'sgd','gnb', 'bnb', 'mlp']
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import SGDClassifier
        from sklearn.naive_bayes import GaussianNB, BernoulliNB
        from sklearn.neural_network import MLPClassifier
        if   classifier == 'svm': self.classifier = SVC(C=10, gamma='auto', kernel='linear',probability=True) 
        elif classifier == 'knn': self.classifier = KNeighborsClassifier(n_neighbors=1) 
        elif classifier == 'lr':  self.classifier = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', max_iter=1000)
        elif classifier == 'sgd': self.classifier = SGDClassifier(loss='log', max_iter=1000) 
        elif classifier == 'gnb': self.classifier = GaussianNB()  
        elif classifier == 'bnb': self.classifier = BernoulliNB() 
        elif classifier == 'mlp': self.classifier = MLPClassifier(hidden_layer_sizes=(5,), max_iter=100)

    def expand_LICP(self, support_set, X_hat, y_hat, way, num_support, pseudo_y, acc_list):
        support_set_l= copy.deepcopy(support_set)
        if acc_list[0]>self.confidence_ratio: step=min(self.step*2,self.step_max) 
        else: step =  max(self.step, 1) 
        
        self.elasticnet = ElasticNet(alpha=1, l1_ratio=0.45, fit_intercept=True, normalize=True, warm_start=True, selection='cyclic')
        _, coefs, dual_gap = self.elasticnet.path(X_hat, y_hat, l1_ratio=0.45)           
        coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[::-1, num_support:, :]), axis=2) 
        
        selected = np.zeros(way)
        for gamma in coefs: 
            for i, g in enumerate(gamma):
                if g == 0.0 and (i+num_support not in support_set_l) and (selected[pseudo_y[i]] < step):
                    support_set_l.append(i+num_support)
                    selected[pseudo_y[i]] += 1
            if np.sum(selected >= step) == way: break
        return support_set_l
    
    def expand_RICP(self, support_set, X_hat, y_hat, way, num_support, pseudo_y, acc_list):
        support_set_r= copy.deepcopy(support_set)
        if acc_list[0]>self.confidence_ratio: step=min(self.step*2,self.step_max)
        else: step =  max(self.step, 1) 
        
        self.elasticnet = ElasticNet(alpha=1, l1_ratio=0.45, fit_intercept=True, normalize=True, warm_start=True, selection='cyclic')
        _, coefs, dual_gap = self.elasticnet.path(X_hat, y_hat, l1_ratio=0.45)           
        coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[::-1, num_support:, :]), axis=2)  
       
        selected = np.zeros(way)
        for gamma in coefs: 
            for i, g in enumerate(gamma):
                if g == 0.0 and (i+num_support not in support_set_r) and (selected[pseudo_y[i]] < step):
                    support_set_r.append(i+num_support)
                    selected[pseudo_y[i]] += 1
            if np.sum(selected >= step) == way: break
        return support_set_r

    def stereopsis(self, support_set, X_hat_l, y_hat_l, X_hat_r, y_hat_r, way, num_support, pseudo_y, acc_list, args):
        support_set_l = self.expand_LICP(support_set, X_hat_l, y_hat_l, way, num_support, pseudo_y[0:len(pseudo_y)//2], acc_list)      
        support_set_r = self.expand_LICP(support_set, X_hat_r, y_hat_r, way, num_support, pseudo_y[len(pseudo_y)//2:], acc_list)     
        
        if args.stereopsis == 'intersection': support_new = list(filter(lambda x: x in support_set_l, support_set_r))  
        elif args.stereopsis == 'union': support_new = list(set(support_set_l).union(set(support_set_r))) 
        elif args.stereopsis == 'le' and args.expand_style=='linear': support_new = list(support_set_l)
        elif args.stereopsis == 're' and args.expand_style=='linear': support_new = list(support_set_r)
        return support_new

    def nerve_conduct(self, X1, X2, y):    
        self.support_X1 = self.norm(X1) 
        self.support_X2 = self.norm(X2) 
        self.support_y = y              
    
    def perception(self, X1, X2, args, unlabel_X1=None, unlabel_X2=None, show_detail=False, query_y=None):  
        assert self.support_X1 is not None
        support_X1, support_X2, support_y = self.support_X1, self.support_X2, self.support_y 
        way, num_support = self.num_class, len(support_X1)      

        query_X1 = self.norm(X1)                    
        if unlabel_X1 is None: unlabel_X1 = query_X1 
        else: unlabel_X1 = self.norm(unlabel_X1)    
        query_X2 = self.norm(X2)                    
        if unlabel_X2 is None: unlabel_X2 = query_X2 
        else: unlabel_X2 = self.norm(unlabel_X2)    
        num_unlabel = unlabel_X1.shape[0]                  
        support_set = np.arange(num_support).tolist() 
        
        
        lembeddings = np.concatenate([support_X1, unlabel_X1]) 
        rembeddings = np.concatenate([support_X2, unlabel_X2]) 
        lX, rX = self.lembed(lembeddings), self.rembed(rembeddings)    
        
        H_l = np.dot(np.dot(lX, np.linalg.inv(np.dot(lX.T, lX))), lX.T)   
        X_hat_l = np.eye(H_l.shape[0]) - H_l   
        H_r = np.dot(np.dot(rX, np.linalg.inv(np.dot(rX.T, rX))), rX.T)   
        X_hat_r = np.eye(H_r.shape[0]) - H_r   

        support_X =  np.concatenate((support_X1, support_X2), axis=0)   
        query_X =  np.concatenate((query_X1, query_X2), axis=0)         
        unlabel_X =  np.concatenate((unlabel_X1, unlabel_X2), axis=0)   
        support_y = np.concatenate((support_y, support_y), axis=0)      
        query_y =  np.concatenate((query_y, query_y), axis=0)          
        
        self.classifier.fit(support_X, support_y)    
        
        if show_detail: 
            acc_list = []
            probs = self.classifier.predict_proba(query_X)
            prob_query, predicts = np.max(probs, 1), np.argmax(probs,1)
            acc_list.append(np.mean(predicts == query_y))  
                    
        probs = self.classifier.predict_proba(unlabel_X)
        prob_unlabel, pseudo_y = np.max(probs, 1), np.argmax(probs,1)
        
        y = np.concatenate([support_y, pseudo_y])       
        Y = self.label2onehot(y, way)                    
        y = np.argmax(Y, axis=1)

        y_l = np.concatenate([support_y[0:support_y.shape[0]//2], pseudo_y[0:pseudo_y.shape[0]//2]])         
        Y_l = self.label2onehot(y_l, way)                    
        y_l = np.argmax(Y_l, axis=1)
        y_r = np.concatenate([support_y[support_y.shape[0]//2:], pseudo_y[pseudo_y.shape[0]//2:]])        
        Y_r = self.label2onehot(y_r, way)                    
        y_r = np.argmax(Y_r, axis=1)
        y_hat_l = np.dot(X_hat_l, Y_l) 
        y_hat_r = np.dot(X_hat_r, Y_r) 
        
        support_set = self.stereopsis(support_set, X_hat_l, y_hat_l, X_hat_r, y_hat_r, way, num_support, pseudo_y, acc_list, args) 

        if args.stereopsis == 'intersection' or args.stereopsis == 'union':
            support_X_new =  np.concatenate((lembeddings[support_set], rembeddings[support_set]), axis=0)
            support_y_new =  np.concatenate((y_l[support_set],y_r[support_set]), axis=0)
            self.classifier.fit(support_X_new, support_y_new)
        elif args.stereopsis == 'le':
            support_X_new = lembeddings[support_set]
            support_y_new = y_l[support_set]
            self.classifier.fit(support_X_new, support_y_new)
        elif args.stereopsis == 're':
            support_X_new = rembeddings[support_set]
            support_y_new = y_r[support_set]
            self.classifier.fit(support_X_new, support_y_new)
            
        probs = self.classifier.predict_proba(query_X)
        prob_, predicts = np.max(probs, 1), np.argmax(probs,1)
        for i in range(len(predicts)//2):
            if predicts[i]!=predicts[i+len(predicts)//2]: 
                if prob_[i]>prob_[i+len(probs)//2]: predicts[i+len(predicts)//2]=predicts[i]
                else: predicts[i]=predicts[i+len(predicts)//2]
            else: predicts[i] = predicts[i]  
        if show_detail:
            acc_list.append(np.mean(predicts == query_y))
            return acc_list

        return predicts
