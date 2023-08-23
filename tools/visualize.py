import time
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import torch
import mmcv
import numpy as np
from shutil import copyfile
import os
import datetime
import cv2
from class_names import get_palette

def main():
    parser = ArgumentParser()
    #parser.add_argument('--img_folder', help='Image file')
    parser.add_argument('--config', default='/media/meiguiz/HIKSEMI/mmsegmentation/work_dirs/FSM_CAL/segformer_mit-b3_8x1_1024x1024_160k_cityscapes.py', help='Config file')
    parser.add_argument('--checkpoint', default='/media/meiguiz/HIKSEMI/mmsegmentation/work_dirs/FSM_CAL/iter_100000.pth', help='Checkpoint file')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='rellis_group',
        help='Color palette used for segmentation map')
    args = parser.parse_args()


    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    input_path = '/media/meiguiz/HIKSEMI/mmsegmentation/data/trosd/c'
    output_path = '/media/meiguiz/HIKSEMI/mmsegmentation/data/trosd/d'
    t = 0
    for file in os.listdir(input_path):
        img_file = os.path.join(input_path, file)

        img = mmcv.imread(img_file)
        t1 = time.time()
        result = inference_segmentor(model, img)
        t2 = time.time() - t1
        vis, seg = show_result_pyplot(model, img_file, result, get_palette(args.palette),)
        print("prossing:", file)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        save_file = os.path.join(output_path, file)
        cv2.imwrite(save_file, vis)
        t += t2
    fps = 1.0 / (t / 3639)
    print('fps:', fps)



if __name__ == '__main__':
    main()
