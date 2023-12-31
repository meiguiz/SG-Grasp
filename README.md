# SG-Grasp
SG-Grasp: Semantic Segmentation Prior Guided Robotic Grasp Oriented to Weakly Textured Objects Based on RGB-D Sensors

> Ling Tong, Kechen Song, Member, IEEE, Hongkun Tian, Yi Man, Yunhui Yan, and Qinggang Meng, Senior Member, IEEE. 

![图片9](https://github.com/meiguiz/SG-Grasp/assets/90629126/11baf3ef-e684-4621-b6d3-5c716be03738)



## SG-Grasp
---
The video of robotic experiments can be found at [this](https://youtu.be/ChjeqFk0_mA). 

Semantic segmentation for reflective and transparent objects on [TROSD](http://www.tsinghua-ieit.com/trosd) dataset [1] 

![image](https://github.com/meiguiz/SG-Grasp/assets/90629126/18c51507-0e33-4e14-88f5-d0306b64baf8)


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

2. Download the provided RTSegNet weights trained on TROSD dataset and put the weight in work_dirs. 
- [weight](https://drive.google.com/file/d/1Ik9ffzRdolUc0g89fnDy8EQT9YG1Xx_d/view?usp=sharing)

3. Download the [TROSD](http://www.tsinghua-ieit.com/trosd) dataset[1] and change the format as follows:
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
4. Set the path to the dataset in config file.


### Train

To train RTSegNet on the TROSD dataset. 
```
$ python tools/train.py configs/rtsegnet/rtsegnet_trosd.py
```


### Evaluation

To evaluate RTSegNet on the TROSD dataset
```
$ python tools/test.py --configs $CONFIG_PATH/rtsegnet_trosd.py \
    --checkpoint $WEIGHT_PATH/iter_120000.pth \
    --eval 
```


### Visualization

To visualize the inference results of RTSegNet on the TROSD dataset. Change the config, checkpoint, input_data and output_data dirs in visualize.py to your path.
```
$ python tools/visualize.py 
```


### Reference
[1] T. Sun, G. Zhang, W. Yang, J.-H. Xue, and G. Wang, "TROSD: A New RGB-D Dataset for Transparent and Reflective Object Segmentation in Practice," IEEE Transactions on Circuits and Systems for Video Technology, 2023.


