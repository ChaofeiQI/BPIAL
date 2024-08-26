from data.sets.miniImageNet import MiniImagenet        # train,val,test版本
from data.sets.tieredImageNet import tieredImageNet
from data.sets.cifarfs import CIFARFS
from data.sets.cub_croped import CUB_Croped
from data.sets.aircraft import AIRCRAFT

__imgfewshot_factory = {
        'MiniImagenet': MiniImagenet,
        'TieredImagenet': tieredImageNet,
        'CIFARFS': CIFARFS,
        'CUB_Croped':CUB_Croped,
        'Aircraft':AIRCRAFT
}


def get_names():
    return list(__imgfewshot_factory.keys()) 


def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgfewshot_factory.keys())))
    return __imgfewshot_factory[name](**kwargs)

