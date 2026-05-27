#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:23:13 2024

@author: des
"""
import torch
import numpy as np
import albumentations as A
import albumentations.pytorch as Ap
from folder_locations import get_results_folder, get_data_folder
import cv2

from nn_detect_browning import LRegressionModule
from nn_detect_browning_proj import LRegressionModule as LRegressionModuleProj

def evaluate_constant_baselines(checkpoint_path, proj_network, augmentations):
    if proj_network:
        l_module = LRegressionModuleProj.load_from_checkpoint(checkpoint_path)
        black_img = augmentations(image=np.zeros((720, 720)))["image"].cuda()[None, ...]
    else:
        l_module = LRegressionModule.load_from_checkpoint(checkpoint_path)
        black_img = augmentations(image=np.zeros((690, 690)))["image"].cuda()[None, ...]
    l_module.freeze()
    l_module.eval()

    black_score = l_module(black_img).cpu().item()
    print(f"black baseline score = {black_score+1:.2f}")

    zeros_img = torch.zeros_like(black_img)
    zero_score = l_module(zeros_img).cpu().item()
    print(f"zero baseline score = {zero_score+1:.2f}")


def evaluate_longitudinal_baseline(ig_path):
    csv_results = np.loadtxt(
        fname = ig_path / "random100_augmentations" / "results.csv",
        #fname = ig_path / "no_augmentations" / "results.csv",
        delimiter = ",",
        skiprows = 1
    )
    in_range_fraction = np.mean(np.logical_and(
        csv_results[:,5] < 3.5,
        csv_results[:,5] > -0.5).astype(float))
    print(f"in range = {in_range_fraction*100}%")

    brown_rows = csv_results[:,2] > 0
    non_brown_rows = csv_results[:,2] == 0

    print(f"Mean baseline value = {np.mean(csv_results[:,5])+1:.2f}, brown apples = {np.mean(csv_results[brown_rows,5])+1:.2f}, non-brown apples = {np.mean(csv_results[non_brown_rows,5])+1:.2f}")
    print(f"std baseline value = {np.std(csv_results[:,5]):.2f}, brown apples = {np.std(csv_results[brown_rows,5]):.2f}, non-brown apples = {np.std(csv_results[non_brown_rows,5]):.2f}")

    non_brown_and_close_to_zero_fraction = np.mean(np.logical_and(
        csv_results[non_brown_rows,5] < 0.5,
        csv_results[non_brown_rows,5] > -0.5).astype(float))
    print(f"non brown and close to zero = {non_brown_and_close_to_zero_fraction*100}%")

    brown_and_increasing_fraction = np.mean((
            csv_results[brown_rows,4] > csv_results[brown_rows,5]
        ).astype(float))
    print(f"brown and increasing = {brown_and_increasing_fraction*100}%")

    non_brown_and_increasing_fraction = np.mean((
            csv_results[non_brown_rows,4] - csv_results[non_brown_rows,5] > 0
        ).astype(float))
    print(f"non-brown and increasing = {non_brown_and_increasing_fraction*100}%")


def evaluate_network(checkpoint_path, ig_path, proj_network, name, augmentations):
    print(f"{name}:")
    evaluate_constant_baselines(checkpoint_path, proj_network, augmentations)
    evaluate_longitudinal_baseline(ig_path)
    print()


if __name__ == "__main__":
    augmentations_ct = A.Compose([
        A.CropAndPad(px=-1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
        A.CropAndPad(px=1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
        A.Normalize(mean=np.float32(0.7961113591168265), std=np.float32(0.14691202875131731), max_pixel_value=1.0),
        Ap.ToTensorV2()
        ])

    augmentations_proj = A.Compose([
        A.Crop(x_min=0, x_max=690, y_min=131, y_max=690-152),
        A.CropAndPad(px=-1, keep_size=False),
        A.CropAndPad(px=1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
        A.Normalize(mean=np.float32(0.54003301858902).item(), std=np.float32(0.573314089958484).item(), max_pixel_value=1.0),
        Ap.ToTensorV2()
        ])

    evaluate_network(
        checkpoint_path = get_results_folder() / "2024-03-08_nn_detect_browning_1_detection_final" / "checkpoints" / "mae_val" / "best_mae_val_epoch=11873.ckpt",
        ig_path = get_results_folder() / "2024-04-19_IG_compare_1_detection_final",
        proj_network = False,
        name = "CT slices detection",
        augmentations = augmentations_ct
        )
    evaluate_network(
        checkpoint_path = get_results_folder() / "2024-03-11_nn_detect_browning_1_prediction_final" / "checkpoints" / "mae_val" / "best_mae_val_epoch=5276.ckpt",
        ig_path = get_results_folder() / "2024-04-18_IG_compare_1_prediction_final",
        proj_network = False,
        name = "CT slices prediction",
        augmentations = augmentations_ct
        )
    evaluate_network(
        checkpoint_path = get_results_folder() / "2024-04-01_nn_detect_browning_proj_optuna_1_detection_final" / "trial_0" / "checkpoints" / "mae_val" / "best_mae_val_epoch=7387_mae_val=0.2954833508.ckpt",
        ig_path = get_results_folder() / "2024-04-20_IG_compare_proj_1_detection_final",
        proj_network = True,
        name = "radiographs detection",
        augmentations = augmentations_proj
        )
    evaluate_network(
        checkpoint_path = get_results_folder() / "2024-04-03_nn_detect_browning_proj_optuna_1_prediction_final" / "trial_0" / "checkpoints" / "mae_val" / "best_mae_val_epoch=490_mae_val=0.8732241988.ckpt",
        ig_path = get_results_folder() / "2024-04-19_IG_compare_proj_1_prediction_final",
        proj_network = True,
        name = "radiographs prediction",
        augmentations = augmentations_proj
        )