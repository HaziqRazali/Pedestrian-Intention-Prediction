Readme not complete

# Pedestrian-Intention-Prediction

We develop a method for pedestrian pose estimation and intent prediction. The source code is built on top of [PifPaf](https://github.com/vita-epfl/openpifpaf/blob/master/README.md) with very little modifications. Work done at EPFL VITA laboratory under Prof. Alexandre Alahi.

# Contents
------------
  * [Requirements](#requirements)
  * [Brief Project Structure](#brief-project-structure)
  * [Results](#results)
  * [Install](#install)
  * [Train](#train)
  * [Test](#test)

# Requirements
------------
What we used to develop the system

  * Python 3
  * PyTorch 1.0.1
  * OpenCV
  * Ubuntu 18.04.2
  
# Results
------------
 
[![Vid](/others/Video.png)](https://www.youtube.com/watch?v=KTmi0D-UTTQ)

# Setup
------------

* Install the source code
```
pip3 install openpifpaf
pip3 install numpy cython
pip3 install --editable '.[train,test]'
```
* Download the [MSCOCO](http://cocodataset.org/#download) 2017 Train images [118K/18GB], 2017 Val images [5K/1GB] and the 2017 Train/Val annotations [241MB] and unzip them in the folders as shown below
* Download the JAAD dataset and place them in
* Run the script to convert the JAAD videos into frames 
* JAAD annotation file

    ├── datasets  
        ├── coco
            ├── annotations 
            ├── images   
                ├── train2017
                ├── val2017
        ├── jaad  

# Train
------------
 
* Download PifPaf's resnet50 model from [openpifpaf's](https://github.com/vita-epfl/openpifpaf) pretrained models or from a [direct link](https://drive.google.com/file/d/1lJCdGLYqWGNDHxFkg1esGZRZ2SzRRbrR/view?usp=sharing) and place it in `outputs/`. Note that the current version only works with resnet50.
* Run `./train.sh` which contains the following command
```
CUDA_VISIBLE_DEVICES="0,1,3" python3 -m openpifpaf.train \
  --pre-lr=1e-5 \
  --lr=1e-5 \
  --momentum=0.95 \
  --epochs=20 \
  --lr-decay 60 70 \
  --jaad_batch_size=3 \
  --batch-size=6 \
  --basenet=resnet50block5 \
  --head-quad=1 \
  --headnets pif paf crm \
  --square-edge=401 \
  --regression-loss=laplace \
  --lambdas 30 2 2 50 3 3 \
  --freeze-base=1 \
  --jaad_train "singletxt_train_3s" --jaad_val "singletxt_val_3s" --jaad_pre_train "singletxt_pre_train_3s"
```

The arguments are as follows
* `CUDA_VISIBLE_DEVICES`: To control which CUDA devices the program should use.
* `pre-lr`: Learning rate when the base net is frozen during the first epoch to initialize the head nets.
* `lr`: Learning rate after the base net is unfrozen.
* `momentum`: Adam parameter.
* `epochs`: Number of epochs to train the model for.
* `lr-decay`: 
* `jaad-batch-size`: Batch size for the JAAD dataset.
* `coco-batch-size`: Batch size for the COCO dataset.
* `basenet`: PifPaf pretrained base network. Currently only works for resnet50.
* `head-quad`: Number of pixel shuffle layers in the head net. Each layer has been hardcoded to upsample the input by a factor of 2 i.e. from H/8,W/8 to H/4,W/4
* `headnets`: Head nets to use. Code does not work with anything other than `pif paf crm`
* `square-edge`: Preprocessing parameter as done in PifPaf.
* `regression-loss`: Loss used for the vector components in PifPaf.
* `lambdas`: Loss weights for Pif's confidence, regression and scale heads and for Paf's confidence and two regression heads.
* `freeze-base`: 1 if the base network should be frozen to initialize the task heads and 0 if not.
* `jaad_train`: Path to JAAD dataset

At any point of time, the training and validation curves can be visualized as follows
```
python3 -m openpifpaf.logs \
  outputs/resnet50block5-pif-paf-edge401-190424-122009.pkl.log \
  outputs/resnet101block5-pif-paf-edge401-190412-151013.pkl.log \
  outputs/resnet152block5-pif-paf-edge401-190412-121848.pkl.log
```

# Test
------------

* Generate the video with the predicted poses by running `./paint_pose.sh` which contains the following command
```
CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.video_pose --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190526-203030.pkl.epoch006
```

* Generate the video with the predicted action activity map by running `./paint_action.sh` which contains the following command
```
CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.video_crm --batch-size 1 --jaad_batch_size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190526-203030.pkl.epoch006 \
```

* Guided backpropagation as shown in the paper
* Evaluate precision and recall
