#############################################################################################
# 超参数设置
# GPU_ID: '0','1','2','3','4','5','6','7'
# DATASET: 'CIFARFS','CUB','FC100','StanfordDog','StanfordCar','MiniImagenet','TieredImagenet' 
#############################################################################################
GPU_IDs='0,1'
DATASET='CIFARFS'
WEIGHTS='1-1-1'

date=`date +"%Y-%m-%d"`
date_time=`date +"%Y-%m-%d-%H:%M"`
MODEL_SAVE_1='ckpt/'$date'/'$DATASET'_'visionres12'_'Weights:$WEIGHTS'/1-shot'
MODEL_SAVE_5='ckpt/'$date'/'$DATASET'_'visionres12'_'Weights:$WEIGHTS'/5-shot'


################
# 1.训练
################
python main.py --dataset $DATASET -g $GPU_IDs --nKnovel 5 --nExemplars 1 --phase val --mode train  --save-dir $MODEL_SAVE_1   # 1 shot
