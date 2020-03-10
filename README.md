Readme not complete

# Pedestrian-Intention-Prediction

We develop a method for pedestrian pose estimation and intent prediction. The source code is built on top of [PifPaf](https://github.com/vita-epfl/openpifpaf/blob/master/README.md) with very little modifications. Work done at [EPFL VITA laboratory](https://www.epfl.ch/labs/vita/) under Professor Alexandre Alahi. Details can be found in the [report in google drive](https://drive.google.com/file/d/194Pfagd9HWF88UZ1kqE9omjutZ9GIC9L/view?usp=sharing).

# Contents
------------
  * [Requirements](#requirements)
  * [Results](#results)
  * [Brief Project Structure](#brief-project-structure)
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

# Brief Project Structure
------------

Incomplete

      ├── datasets                        : Where the datasets are stored. See setup for more details.
      ├── Pedestrian-Intention-Prediction : Cloned project root
          ├── openpifpaf      
              ├── decoder                 : Scripts to decode the Pif and Paf fields into skeletons.
              ├── encoder                 : Scripts to preprocess the ground truth for the Pif and Paf heads.              
              ├── network                 : Scripts to build the base and head encoder networks.                
              ├── datasets.py             : Script containing the data loaders.          
              ├── logs.py                 : Script to generate the training and validation curves.
              ├── train.py                : Script to train the model.
              ├── paint_action.py         : Script to to generate the video with the predicted activity map.
              ├── paint_pose.py           : Script to to generate the video with the predicted poses.

          ├── paint_action.sh             : Shell script that runs paint_action.py
          ├── paint_pose.sh               : Shell script that runs paint_pose.py
          ├── train.sh                    : Shell script that runs train.py

          ├── outputs     : where the models and logs are stored  
          ├── plots (check)
          ├── scripts (check)
          ├── tests (check) 
          ├── Report.pdf          

# Setup
------------

* Create anaconda environment and install PyTorch and OpenCV
```
conda create -n env python=3.6 anaconda
conda activate env
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install opencv-python
```

* Install the source code
```
pip install Pedestrian-Intention-Prediction
cd Pedestrian-Intention-Prediction
pip install --editable '.[train,test]'
```
* Download and unzip the [MSCOCO](http://cocodataset.org/#download) `2017 Train images [118K/18GB]` and `2017 Val images [5K/1GB]` into the `images` folder the `2017 Train/Val annotations [241MB]` into the `annotations` folder.
* Download the JAAD clips (UNRESIZED) and unzip them in the `videos` folder.
* Run the script `split_clips_to_frames.sh` to convert the JAAD videos into frames. Each frame will be placed in a folder under the `scene` folder. Note that this takes 169G of space.
* Download and unzip the JAAD annotation files into the `annotations` folder.

How the datasets folder should look like

      ├── datasets  
          ├── coco
              ├── annotations 
                  ├── captions_train2017.json
              ├── images   
                  ├── train2017
                      ├── 000000000009.jpg
                  ├── val2017
                      ├── 000000000139.jpg
                      
          ├── jaad  
              ├── annotations 
                  ├── singletxt_train_1s
                      ├── train.txt
                  ├── singletxt_val_1s
                      ├── val.txt
              ├── videos 
                  ├── 0001.mp4
              ├── scene 
                  ├── 0001
                      ├── 0001.png

# Train
------------
 
* Download PifPaf's resnet50 model from a [direct link](https://drive.google.com/open?id=13HPZdbdjg09paa-RMTpMPdm5nmgT1jYF) or from [openpifpaf's](https://github.com/vita-epfl/openpifpaf) pretrained models, rename it to `resnet50block5-pif-paf-edge401.pkl` and place it in `outputs/`. Note that the current version only works with resnet50.
* Run `./train.sh` which contains the following command
```
CUDA_VISIBLE_DEVICES="0,1,3" python3 -m openpifpaf.train \
  --pre-lr=1e-5 \
  --lr=1e-5 \
  --momentum=0.95 \
  --epochs=20 \
  --lr-decay 60 70 \
  --jaad-batch-size=3 \
  --coco-batch-size=6 \
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
* `head-quad`: Number of [pixel shuffle](https://pytorch.org/docs/stable/nn.html#torch.nn.PixelShuffle) layers in the head net. Each layer has been hardcoded to upsample the input by a factor of 2 i.e. from `H/8,W/8` to `H/4,W/4`
* `headnets`: Head nets to use. Code does not work with anything other than `pif paf crm`
* `square-edge`: Preprocessing parameter as done in PifPaf.
* `regression-loss`: Loss used for the vector components in PifPaf.
* `lambdas`: Loss weights for Pif's confidence, regression and scale heads and for Paf's confidence and two regression heads.
* `freeze-base`: 1 if the base network should be frozen to initialize the task heads and 0 if not.
* `jaad_train`: Path to JAAD dataset

At any point of time, the training and validation curves can be visualized as follows
```
python3 -m openpifpaf.logs \
  outputs/model_name.pkl.log \
```

# Test
------------

* Either use the model you trained or download a trained one [here](https://drive.google.com/file/d/1SWY2GDEyQp-wNmmFKMBTZHSYJN7HKfgU/view?usp=sharing) and place it in the `outputs` folder.

* Generate the video with the predicted poses by running `./paint_pose.sh` which contains the following command
```
CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.video_pose --batch-size 1 --jaad-batch-size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190525-002719.pkl.epoch006
```

* Generate the video with the predicted action activity map by running `./paint_action.sh` which contains the following command
```
CUDA_VISIBLE_DEVICES="0" python3 -m openpifpaf.video_crm --batch-size 1 --jaad-batch-size 1 \
--checkpoint outputs/resnet50block5-pif-paf-crm-edge401-190525-002719.pkl.epoch006
```

* Generate the video showing the results of guided backpropagation by running
* Evaluate precision and recall
