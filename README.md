# SG-Grasp
SG-Grasp: Semantic Segmentation Prior Guided Robotic Grasp Oriented to Weakly Textured Objects Based on RGB-D Sensors

> Ling Tong, Kechen Song, Member, IEEE, Hongkun Tian, Yi Man, Yunhui Yan, and Qinggang Meng, Senior Member, IEEE. 

![) M52~1LS2M75{9S}(@L5 S](https://github.com/meiguiz/SG-Grasp/assets/90629126/317054b6-36a6-4a57-a31f-c5322974de76)

## SG-Grasp
---
The video of robotic experiments can be found at [this](https://youtu.be/ChjeqFk0_mA). 

Semantic segmentation for reflective and transparent objects on [TROSD](http://www.tsinghua-ieit.com/trosd) dataset [1] 

![69TVW{7S)BX@QVVA}J`LODD](https://github.com/meiguiz/SG-Grasp/assets/90629126/1c8d6ee4-1ac0-4b87-b32a-1f5d4af4a466)

## Getting Started
---
### Environment Setup

1. Setup anaconda environment
```
$ conda create --name sggrasp python=3.8 -y
$ conda activate sggrasp
$ conda install pytorch torchvision -c pytorch
$ pip install -U openmim
$ mim install mmengine
$ mim install "mmcv>=2.0.0"
$ git clone -b main https://github.com/meiguiz/SG-Grasp.git
$ cd SG-Grasp
$ pip install -v -e .
```

2. Download the provided RTSegNet weights pre-trained on TROSD dataset. 
- [rgb only](https://drive.google.com/file/d/128PI9Z6h3VBjBOVEIHV6lPUnk9YerfL6/view?usp=sharing)

3. Download the WISDOM-Real dataset [[Link]](https://sites.google.com/view/wisdom-dataset/dataset_links)
 ```
   data/trosd
   ├─TR_annotations
   │  ├─cleargrasp_real_known
   │  │  └─cg_real_test_d415_000000000_1_v_group6.png
   │  ├─cleargrasp_real_novel
   │  │  └─cg_real_val_d435_000000000_1_v_group6.png
   │  └─Trosd
   │  │  └─new_room_3_group6.png
   ├─TR-with-annotations
   │  ├─cleargrasp_real_known
   │  │  └─cg_real_test_d415_000000000_1_v.png
   │  ├─cleargrasp_real_novel
   │  │  └─cg_real_val_d435_000000000_1_v.png
   │  └─Trosd
   │  │  └─new_room_3.png
   ├─test_ours.txt
   ├─val_ours.txt
   ├─train_ours.txt
   ├─val_cleargrasp_known.txt
   ├─val_cleargrasp_novel.txt
   │  
   ├─other_files
   ```
4. Set the path to the dataset and pretrained weights (You can put this into your bash profile)
```
$ export WISDOM_PATH={/path/to/the/wisdom-real/high-res/dataset}
$ export WEIGHT_PATH={/path/to/the/pretrained/weights}

```



### Train

To train an SF Mask R-CNN (confidence fusion, RGB-noisy depth as input) on a synthetic dataset. 
```
$ python train.py --gpu 0 --cfg rgb_noisydepth_confidencefusion
```
To fine-tune the SF Mask R-CNN on WISDOM dataset
```
$ python train.py --gpu 0 --cfg rgb_noisydepth_confidencefusion_FT --resume
```

### Evaluation

To evaluate an SF Mask R-CNN (confidence fusion, RGB-noisy depth as input) on a WISDOM dataset
```
$ python eval.py --gpu 0 --cfg rgb_noisydepth_confidencefusion \
    --eval_data wisdom \
    --dataset_path $WISDOM_PATH \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_ConfidenceFusion.tar 
```


### Visualization

To visualize the inference results of SF Mask R-CNN on a WISDOM dataset
```
$ python inference.py --gpu 0 --cfg rgb_noisydepth_confidencefusion \
    --eval_data wisdom --vis_depth \
    --dataset_path $WISDOM_PATH \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_ConfidenceFusion.tar 
```

<img src="./imgs/example1.png" height="150">
<img src="./imgs/example2.png" height="150">
<img src="./imgs/example3.png" height="150">


### Demo with RealSense

To run real-time demo with realsense-d435
```
# SF Mask R-CNN (confidence fusion)
$ python demo.py --cfg rgb_noisydepth_confidencefusion \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_ConfidenceFusion.tar 

# SF Mask R-CNN (early fusion)
$ python demo.py --cfg rgb_noisydepth_earlyfusion \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_EarlyFusion.tar 


# SF Mask R-CNN (late fusion)
$ python demo.py --cfg rgb_noisydepth_latefusion \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_LateFusion.tar 
```

## Authors
* **Seunghyeok Back** [seungback](https://github.com/SeungBack)
* **Raeyoung Kang** [raeyo](https://github.com/raeyo)
* **Taewon Kim** [ailabktw](https://github.com/ailabktw)
* **Joosoon Lee** [joosoon](https://github.com/joosoon)


## Citation
If you use our work in a research project, please cite our work:
```
[1] @inproceedings{back2020segmenting,
  title={Segmenting unseen industrial components in a heavy clutter using rgb-d fusion and synthetic data},
  author={Back, Seunghyeok and Kim, Jongwon and Kang, Raeyoung and Choi, Seungjun and Lee, Kyoobin},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  pages={828--832},
  year={2020},
  organization={IEEE}
}
```

