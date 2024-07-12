#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:16:02 2024

@author: des
"""
from pathlib import Path
import os
import sys
import torch
import tifffile
import numpy as np
import scipy
import pytorch_lightning as pl
import albumentations as A
import albumentations.pytorch as Ap
import itertools
from matplotlib import pyplot as plt
import cv2

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from nn_detect_browning import parse_metadata, load_slice_centered, LRegressionModule, parse_core_slice_nrs
from invertible_augmentations import generate_flip_rot90_augmentations

if __name__ == "__main__":
    base_path = get_data_folder()
    recons_path = base_path / "recons_bh_corr_registered_crop"
    masks_path = base_path / "recons_bh_corr_registered_crop_dm"
    metadata_path = base_path / "Brae_browning_score.csv"
    centers_of_mass_path = base_path / "centers_of_mass.csv"
    
    if sys.argv[1] == "d":
        checkpoint_path = get_results_folder() / "2024-03-08_nn_detect_browning_1_detection_final" / "checkpoints" / "mae_val" / "best_mae_val_epoch=11873.ckpt"
        
        selected_days = [
            "2023-01-23 CA storage 10 weeks out 2 weeks",
            "2023-01-30 CA storage 10 weeks out 3 weeks",
            "2023-02-13 CA storage 13 weeks out day 15",
            "2023-03-13 CA storage 18 weeks out day 8",
            "2023-03-20 CA storage 18 weeks out day 15"
        ]
        day_dict = None
        
    elif sys.argv[1] == "p":
        checkpoint_path = get_results_folder() / "2024-03-11_nn_detect_browning_1_prediction_final" / "checkpoints" / "mae_val" / "best_mae_val_epoch=5276.ckpt"
        
        selected_days = [
            "2023-01-23 CA storage 10 weeks out 2 weeks",
            "2023-02-13 CA storage 13 weeks out day 15",
            "2023-03-20 CA storage 18 weeks out day 15"
        ]
        day_dict = {
            "2023-01-23 CA storage 10 weeks out 2 weeks" : "2023-01-09 CA storage 10 weeks",
            "2023-02-13 CA storage 13 weeks out day 15" : "2023-01-30 CA storage 13 weeks out day 1",
            "2023-03-20 CA storage 18 weeks out day 15" : "2023-03-06 CA storage 18 weeks out day 1"
        }
    
    train_set_nrs = [41, 42, 50, 49, 16, 19, 24, 28, 33, 45, 51, 53, 54, 55, 56,
                     57, 58, 59, 60, 61, 63, 6, 29, 64, 66, 68, 72, 74, 77, 80,
                     81, 82, 85, 22, 34, 35, 46, 67, 70, 71, 73, 75, 76, 78, 79]
    val_set_nrs = [44, 40, 20, 26, 52, 47, 62, 83, 84, 48, 69, 65, 86, 21, 39]
    test_set_nrs = [43, 13, 17, 8, 18, 4, 25, 36, 37, 32, 23, 38, 14, 3, 27]
    
    # Calculated over all included (close to the core) slices of the training set
    dataset_mean = np.float32(0.7961113591168265)
    dataset_std = np.float32(0.14691202875131731)
    region_size = 0.2
    
    core_slice_nrs = parse_core_slice_nrs(centers_of_mass_path)
    scans = parse_metadata(metadata_path, recons_path, core_slice_nrs, selected_days, test_set_nrs, day_dict)
    augmentations = A.Compose([
        A.CropAndPad(px=-1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
        A.CropAndPad(px=1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
        A.Normalize(mean=dataset_mean, std=dataset_std, max_pixel_value=1.0),
        Ap.ToTensorV2()
        ])
    
    l_module = LRegressionModule.load_from_checkpoint(checkpoint_path)
    l_module.freeze()
    l_module.eval()
    
    separate_mae = []
    separate_accuracy = []
    combined_mae = []
    combined_accuracy = []
    for scan in scans:
        min_slice_nr = round(scan.center_slice-(region_size/2*scan.num_slices))
        max_slice_nr = round(scan.center_slice+(region_size/2*scan.num_slices))
        
        scan_results = []
        
        for slice_nr in range(min_slice_nr, max_slice_nr):
            recon_slice_path = scan.path / f"output{slice_nr:05d}.tif"
            mask_slice_path = next((masks_path / str(scan.apple_nr)).iterdir()) / recon_slice_path.name
            img = load_slice_centered((690, 690), recon_slice_path, mask_slice_path, 0)
            
            for inv_augmentation in generate_flip_rot90_augmentations(img.shape):
                img_aug = augmentations(image=inv_augmentation.apply(img))["image"]
                slice_result = l_module(img_aug[None,:,:,:].cuda())[0, 0].cpu().numpy()
                
                slice_score = np.clip(np.round(slice_result), 0, 3)
                separate_mae.append(abs(slice_score-scan.label))
                separate_accuracy.append(float((slice_score>0)^(scan.label==0)))
                scan_results.append(slice_result)
        
        scan_score = np.clip(round(np.median(scan_results)), 0, 3)
        combined_mae.append(abs(scan_score-scan.label))
        combined_accuracy.append(float((scan_score>0)^(scan.label==0)))
        print(f"apple nr = {scan.apple_nr}, scan label = {scan.label}, output = {scan_score}, MAE = {combined_mae[-1]}, browning (yes/no) accuracy = {combined_accuracy[-1]}")
        
    print(f"Combined MAE = {np.mean(combined_mae)}, combined browning (yes/no) accuracy = {np.mean(combined_accuracy)}")
    print(f"Separate MAE = {np.mean(separate_mae)}, separate browning (yes/no) accuracy = {np.mean(separate_accuracy)}")
    
