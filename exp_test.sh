#############################################################################################
# 超参数设置
# GPU_ID: '0','1','2','3','4','5','6','7'
# DATASET: 'CIFARFS','CUB_Croped','Aircraft','CUB','FC100','StanfordDog','StanfordCar','MiniImagenet','TieredImagenet' 
#############################################################################################
GPU_IDs='0'
DATASET='CIFARFS'
WEIGHTS='1-1-1'

date=`date +"%Y-%m-%d"`
date_time=`date +"%Y-%m-%d-%H:%M"`
MODEL_SAVE_1='ckpt/'$date'/'$DATASET'_'visionres12'_'Weights:$WEIGHTS'/1-shot'
MODEL_SAVE_5='ckpt/'$date'/'$DATASET'_'visionres12'_'Weights:$WEIGHTS'/5-shot'
MODEL_PATH_1='ckpt/'$date'/'$DATASET'_'visionres12'_'Weights:$WEIGHTS'/1-shot/best_model.pth.tar'
MODEL_PATH_5='ckpt/'$date'/'$DATASET'_'visionres12'_'Weights:$WEIGHTS'/5-shot/best_model.pth.tar'
SAVE_PATH_1_0='ckpt/'$date'/'$DATASET'_'visionres12'_'Weights:$WEIGHTS'/1-shot/test_1shot/'$date_time''


################
# 2.测试
################
python main.py --dataset $DATASET -g $GPU_IDs --nKnovel 5 --nExemplars 1 --phase test --mode test  --resume $MODEL_PATH_1 --save-dir $SAVE_PATH_1_0  # 1 shot
