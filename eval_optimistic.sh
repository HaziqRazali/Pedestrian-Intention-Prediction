CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.eval_optimistic --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190526-203030.pkl.epoch006 \
--truncate 0 \
--final_frame_offset 0 \
--result "trained1s.txt" \
--jaad_train "singletxt_pre_train_0s" --jaad_val "singletxt_val_0s" --jaad_pre_train "singletxt_pre_train_0s"

CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.eval_optimistic --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190527-174900.pkl.epoch006 \
--truncate 0 \
--final_frame_offset 0 \
--result "trained2s.txt" \
--jaad_train "singletxt_pre_train_0s" --jaad_val "singletxt_val_0s" --jaad_pre_train "singletxt_pre_train_0s"