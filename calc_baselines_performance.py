#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:23:13 2024

@author: des
"""
import torch
import numpy as np
from folder_locations import get_results_folder, get_data_folder

from nn_detect_browning import LRegressionModule
from nn_detect_browning_proj_optuna import LRegressionModule as LRegressionModuleProj

def evaluate_constant_baselines(checkpoint_path, proj_network):
    if proj_network:
        l_module = LRegressionModuleProj.load_from_checkpoint(checkpoint_path)
        dataset_mean = 0.54003301858902
    else:
        l_module = LRegressionModule.load_from_checkpoint(checkpoint_path)
        dataset_mean = 0.7961113591168265
    l_module.freeze()
    l_module.eval()
    
    zeros_img = torch.zeros((1, 1, 690, 690), device="cuda")
    zero_score = l_module(zeros_img).cpu()
    print(f"zero baseline score = {zero_score}")
    black_img = torch.full((1, 1, 690, 690), -dataset_mean, device="cuda")
    black_score = l_module(black_img).cpu()
    print(f"black baseline score = {black_score}")
    
def evaluate_longitudinal_baseline(ig_path):
    csv_results = np.loadtxt(
        fname = ig_path / "random100_augmentations" / "results.csv",
        delimiter = ",",
        skiprows = 1
    )
    in_range_fraction = np.mean(np.logical_and(
        csv_results[:,5] < 3.5,
        csv_results[:,5] > -0.5).astype(float))
    print(f"in range = {in_range_fraction*100}%")
    
    brown_rows = csv_results[:,2] > 0
    non_brown_rows = csv_results[:,2] == 0
    
    print(f"Mean baseline value = {np.mean(csv_results[:,5])}, brown apples = {np.mean(csv_results[brown_rows,5])}, non-brown apples = {np.mean(csv_results[non_brown_rows,5])}")
    print(f"std baseline value = {np.std(csv_results[:,5])}, brown apples = {np.std(csv_results[brown_rows,5])}, non-brown apples = {np.std(csv_results[non_brown_rows,5])}")
    
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
    

def evaluate_network(checkpoint_path, ig_path, proj_network, name):
    print(f"{name}:")
    evaluate_constant_baselines(checkpoint_path, proj_network)
    evaluate_longitudinal_baseline(ig_path)
    print()
    

if __name__ == "__main__":
    evaluate_network(
        checkpoint_path = get_results_folder() / "2024-03-08_nn_detect_browning_1_detection_final" / "checkpoints" / "mae_val" / "best_mae_val_epoch=11873.ckpt",
        ig_path = get_results_folder() / "2024-04-19_IG_compare_1_detection_final",
        proj_network = False,
        name = "CT slices detection"
        )
    evaluate_network(
        checkpoint_path = get_results_folder() / "2024-03-11_nn_detect_browning_1_prediction_final" / "checkpoints" / "mae_val" / "best_mae_val_epoch=5276.ckpt",
        ig_path = get_results_folder() / "2024-04-18_IG_compare_1_prediction_final",
        proj_network = False,
        name = "CT slices prediction"
        )
    evaluate_network(
        checkpoint_path = get_results_folder() / "2024-04-01_nn_detect_browning_proj_optuna_1_detection_final" / "trial_0" / "checkpoints" / "mae_val" / "best_mae_val_epoch=7387_mae_val=0.2954833508.ckpt",
        ig_path = get_results_folder() / "2024-04-20_IG_compare_proj_1_detection_final",
        proj_network = True,
        name = "radiographs detection"
        )
    evaluate_network(
        checkpoint_path = get_results_folder() / "2024-04-03_nn_detect_browning_proj_optuna_1_prediction_final" / "trial_0" / "checkpoints" / "mae_val" / "best_mae_val_epoch=490_mae_val=0.8732241988.ckpt",
        ig_path = get_results_folder() / "2024-04-19_IG_compare_proj_1_prediction_final",
        proj_network = True,
        name = "radiographs prediction"
        )
        
    
    
    
    
    