# DMAP: Decoupling-driven Multi-level Attribute Parsing for Interpretable Outfit Collocation

This is the code for the paper: "DMAP: Decoupling-driven Multi-level Attribute Parsing for Interpretable Outfit Collocation".

## Overview Framework

![image-20240115160920333](./framework.png)

## Project introduction

1. data:
   The required data files, including:
   train/valid/test_list.csv : The training/validate/testing files for the Outfit Compatibility Prediction (OCP) task.
   test_fitb_p/n1/n2/n3.csv : The testing files for the  Fill-in-the-Blank (FITB)  task.
   item_img_num.csv : Attribute information obtained by statistics on IQON 3000 dataset.
   Please download "data" folder from: "https://pan.baidu.com/s/1Bw_4iUnZlMP0mbMGW7dKwg"(Extraction code：wu6r) and put it in the "DMAP" folder.

3. Comp:
    The source code and pre-trained parameters of the PS-OCM.
    `python train.py --batch_size 16 --epoch_num 100 --imgpath (IQON3000 data path)`
   The model training parameters "model.pt" are saved in BaiduNetwork disk. Please download the "result" folder from "https://pan.baidu.com/s/1Bw_4iUnZlMP0mbMGW7dKwg"(Extraction code：wu6r) and put it in the "Comp" folder.

5. FITB:
    The test code for the FITB task.
    `python compute_fitb.py --imgpath (IQON3000 data path)`
