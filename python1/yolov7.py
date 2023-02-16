#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:19:43 2023

@author: intern
"""

%cd /media/intern/6bf406ea-c22f-457e-bdbc-47701a7188a4/home/content/yolov7


import train
import test1 as test
import detect
#%%
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

!git clone https://github.com/WongKinYiu/yolov7
%cd yolov7
torch.cuda.empty_cache()
set PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb:32
#%%
!python train.py --batch 4 --epochs 150 --data ./olddata/data.yaml  --weights "yolov7.pt" --cfg cfg/training/yolov7.yaml --device 0 --hyp data/hyp.scratch.p5.yaml --name yolov
#%%
!python detect.py --weights ./runs/train/yolo8/weights/best.pt --conf 0.3 --source ./olddata/test/images --name yolo_det --save-txt --nosave
#%%
!python test.py --weights runs/train/yolo8/weights/best.pt --data ./olddata/data.yaml --task test --name yolo_det 
#%%
test.test("./olddata/data.yaml")
#%%
test.test("/media/intern/6bf406ea-c22f-457e-bdbc-47701a7188a4/home/content/yolov7/olddata/data.yaml",
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=1,
         dataloader=None,
         save_dir=Path('/media/intern/6bf406ea-c22f-457e-bdbc-47701a7188a4/home/content/yolov7/runs'),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False)
#%%
python test.py --weights runs/train/yolo8/weights/best.pt --data ./olddata/data.yaml --task test --name yolo_det
