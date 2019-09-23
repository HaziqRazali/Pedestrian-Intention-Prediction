
CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.eval --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190525-002719.pkl.epoch006 \
--truncate 0 \
--final_frame_offset 0 \
--result "trained0s_predict0s_truncated.txt" \
--jaad_train "singletxt_pre_train_0s_truncated" --jaad_val "singletxt_val_0s_truncated" --jaad_pre_train "singletxt_pre_train_0s_truncated"

CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.eval --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190525-002719.pkl.epoch006 \
--truncate 0 \
--final_frame_offset 0 \
--result "trained0s_predict1s_truncated.txt" \
--jaad_train "singletxt_pre_train_1s_truncated" --jaad_val "singletxt_val_1s_truncated" --jaad_pre_train "singletxt_pre_train_1s_truncated"

CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.eval --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190525-002719.pkl.epoch006 \
--truncate 0 \
--final_frame_offset 0 \
--result "trained0s_predict2s_truncated.txt" \
--jaad_train "singletxt_pre_train_2s_truncated" --jaad_val "singletxt_val_2s_truncated" --jaad_pre_train "singletxt_pre_train_2s_truncated"

CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.eval --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190526-203030.pkl.epoch006 \
--truncate 0 \
--final_frame_offset 0 \
--result "trained1s_predict0s_truncated.txt" \
--jaad_train "singletxt_pre_train_0s_truncated" --jaad_val "singletxt_val_0s_truncated" --jaad_pre_train "singletxt_pre_train_0s_truncated"

CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.eval --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190526-203030.pkl.epoch006 \
--truncate 0 \
--final_frame_offset 0 \
--result "trained1s_predict1s_truncated.txt" \
--jaad_train "singletxt_pre_train_1s_truncated" --jaad_val "singletxt_val_1s_truncated" --jaad_pre_train "singletxt_pre_train_1s_truncated"

CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.eval --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190526-203030.pkl.epoch006 \
--truncate 0 \
--final_frame_offset 0 \
--result "trained1s_predict2s_truncated.txt" \
--jaad_train "singletxt_pre_train_2s_truncated" --jaad_val "singletxt_val_2s_truncated" --jaad_pre_train "singletxt_pre_train_2s_truncated"