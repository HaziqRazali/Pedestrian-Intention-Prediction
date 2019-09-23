CUDA_VISIBLE_DEVICES="1" python3 -m openpifpaf.eval_crm_recall --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190526-203030.pkl.epoch006 \
--truncate 0 \
--final_frame_offset 0 \
--result "bbox-recall-crm.txt" \
--jaad_train "singletxt_pre_train_1s" --jaad_val "singletxt_val_1s" --jaad_pre_train "singletxt_pre_train_1s"