
# For jaad dataset
CUDA_VISIBLE_DEVICES=None python3 -m openpifpaf.predict -o ./im/ --seed-threshold 0.1 --instance-threshold 0.1 --checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190525-002719.pkl.epoch006 /data/haziq-data/jaad/scene/0301_temp/*.png

# For coco dataset
#CUDA_VISIBLE_DEVICES=None python3 -m openpifpaf.predict -o ./im/ --debug-pif-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --debug-paf-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190520-121617.pkl.epoch032 /data/haziq-data/jaad/scene/0301_temp/*.png

#/data/haziq-data/temp/*.jpg

# --debug-pif-indices 0 1 2 3 4 5 6 7 
# --debug-paf-indices 0 1 2 3 4 5 6 7 