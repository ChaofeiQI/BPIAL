from __future__ import absolute_import
import torch

def init_optimizer(optim, model, lr, weight_decay):
    if optim == 'adam':
        # return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    elif optim == 'amsgrad':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif optim == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)   
    elif optim == 'mixed':  # 2023-11-6-00:31
        # 创建不同的优化器和超参数
        optimizer1 = torch.optim.SGD(model.lbase.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        optimizer2 = torch.optim.SGD(model.lclasifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        optimizer3 = torch.optim.Adam(model.rbase.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer4 = torch.optim.Adam(model.rclasifier.parameters(), lr=lr, weight_decay=weight_decay)
        # 关联优化器和参数组
        optimizer = torch.optim.SGD([
            {'params': model.lbase.parameters(), 'optimizer': optimizer1},
            {'params': model.lclasifier.parameters(), 'optimizer': optimizer2},
            {'params': model.rbase.parameters(), 'optimizer': optimizer3},
            {'params': model.rclasifier.parameters(), 'optimizer': optimizer4}
        ], lr=lr)        
        return optimizer
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optimizer: {}".format(optim))

# 2023-11-6-00:31

# # 按照不同结构将参数分组
# optimizer = optim.SGD([
#     {'params': model.layer1.parameters(), 'lr': 0.01, 'momentum': 0.9},
#     {'params': model.layer2.parameters(), 'lr': 0.001, 'weight_decay': 0.01}
# ], lr=0.1)

# # 创建不同的优化器和超参数
# optimizer1 = optim.SGD(model.layer1.parameters(), lr=0.01, momentum=0.9)
# optimizer2 = optim.Adam(model.layer2.parameters(), lr=0.001, weight_decay=0.01)

# # 关联优化器和参数组
# optimizer = optim.SGD([
#     {'params': model.layer1.parameters(), 'optimizer': optimizer1},
#     {'params': model.layer2.parameters(), 'optimizer': optimizer2}
# ], lr=0.1)