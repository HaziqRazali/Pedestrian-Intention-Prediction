
CUDA_VISIBLE_DEVICES=None python3 -m openpifpaf.video_crm --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190526-203030.pkl.epoch006 \
--jaad_train "singletxt_pre_train_1s" --jaad_val "singletxt_val_1s" --jaad_pre_train "singletxt_pre_train_1s"