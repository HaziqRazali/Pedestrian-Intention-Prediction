
# For jaad dataset
#CUDA_VISIBLE_DEVICES=None python3 -m openpifpaf.predict -o ./im/ --debug-pif-indices 0 1 2 3 4 5 6 7 --checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190519-200218.pkl.epoch030 /data/haziq-data/jaad/scene/0301_temp/*.png

# For coco dataset
CUDA_VISIBLE_DEVICES=None python3 -m openpifpaf.predict -o ./im/ --checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190519-200218.pkl.epoch030 /data/haziq-data/temp/*.jpg

# --debug-pif-indices 0 1 2 3 4 5 6 7 
# --debug-paf-indices 0 1 2 3 4 5 6 7 