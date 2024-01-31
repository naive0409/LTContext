python run_net.py \
  --cfg configs/Thumos/LTContext.yaml \
  DATA.PATH_TO_DATA_DIR /mnt/DataDrive164/zhanghao/datasets/thumos14_lite/actionformer_thumos \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.DATASET 'thumos' \
  TEST.CHECKPOINT_PATH checkpoints/9/checkpoint_epoch_00150.pyth \
  TEST.SAVE_RESULT_PATH /home/ubuntu/users/zhanghao/LTContext/result
#  checkpoints/4/checkpoint_epoch_00150.pyth
# TEST.CHECKPOINT_PATH checkpoints/3/best_checkpoint.pyth \
