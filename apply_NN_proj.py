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
from nn_detect_browning_proj import parse_metadata_proj, LRegressionModule
from invertible_augmentations import generate_flip_augmentations, generate_no_augmentations

if __name__ == "__main__":
    base_path = get_data_folder()
    ff_proj_path = base_path / "projections_flat_crop"
    metadata_path = base_path / "Brae_browning_score.csv"
    experiments_path = get_results_folder()

    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())

    arg_str = sys.argv[1]
    if len(arg_str) == 1:
        arg_str = arg_str + arg_str

    if arg_str[1] == "d":
        checkpoint_path = experiments_path / "2024-04-01_nn_detect_browning_proj_optuna_1_detection_final" / "trial_0" / "checkpoints" / "mae_val" / "best_mae_val_epoch=7387_mae_val=0.2954833508.ckpt"
    elif arg_str[1] == "p":
        checkpoint_path = experiments_path / "2024-04-03_nn_detect_browning_proj_optuna_1_prediction_final" / "trial_0" / "checkpoints" / "mae_val" / "best_mae_val_epoch=490_mae_val=0.8732241988.ckpt"

    if arg_str[0] == "d":
        selected_days = [
            "2023-01-23 CA storage 10 weeks out 2 weeks",
            "2023-01-30 CA storage 10 weeks out 3 weeks",
            "2023-02-13 CA storage 13 weeks out day 15",
            "2023-03-13 CA storage 18 weeks out day 8",
            "2023-03-20 CA storage 18 weeks out day 15"
        ]
        day_dict = None

    elif arg_str[0] == "p":
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

    # Calculated over all projections over all days of the apples of the training set
    dataset_mean = 0.54003301858902
    dataset_std = 0.573314089958484

    l_module = LRegressionModule.load_from_checkpoint(checkpoint_path)
    l_module.freeze()
    l_module.eval()
    print(l_module.trial.params)

    crop_top = l_module.trial.params["crop_top"]
    crop_bottom = l_module.trial.params["crop_bottom"]

    labeled_pstacks = parse_metadata_proj(metadata_path, ff_proj_path, selected_days, test_set_nrs, day_dict)
    augmentations = A.Compose([
        A.Crop(x_min=0, x_max=690, y_min=crop_top, y_max=690-crop_bottom),
        A.CropAndPad(px=-1, keep_size=False),
        A.CropAndPad(px=1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
        A.Normalize(mean=dataset_mean, std=dataset_std, max_pixel_value=1.0),
        Ap.ToTensorV2()
        ])

    separate_mae = []
    separate_accuracy = []
    combined_mae = []
    combined_accuracy = []
    all_score_outputs = []
    for pstack in labeled_pstacks:
        projs = ceu.load_stack(pstack.path, prefix="output")
        stack_results = []

        for i in range(projs.shape[0]):
            img = projs[i]
            for inv_augmentation in generate_no_augmentations():
                img_aug = augmentations(image=inv_augmentation.apply(img))["image"]
                proj_result = l_module(img_aug[None,:,:,:].cuda())[0, 0].cpu().numpy().item()

                proj_score = np.clip(round(proj_result), 0, 3)
                separate_mae.append(abs(proj_score-pstack.label))
                separate_accuracy.append(float((proj_score>0)^(pstack.label==0)))
                stack_results.append(proj_result)
                all_score_outputs.append(proj_score)

        stack_score = np.clip(round(np.median(stack_results)), 0, 3)
        combined_mae.append(abs(stack_score-pstack.label))
        combined_accuracy.append(float((stack_score>0)^(pstack.label==0)))
        print(f"apple nr = {pstack.apple_nr}, scan label = {pstack.label}, output = {stack_score}, MAE = {combined_mae[-1]}, browning (yes/no) accuracy = {combined_accuracy[-1]}")

    print(f"Combined MAE = {np.mean(combined_mae):.2f} ± {np.std(combined_mae):.2f}, combined browning (yes/no) accuracy = {np.mean(combined_accuracy)*100:.1f} ± {np.std(combined_accuracy)*100:.1f}")
    print(f"Separate MAE = {np.mean(separate_mae):.2f} ± {np.std(separate_mae):.2f}, separate browning (yes/no) accuracy = {np.mean(separate_accuracy)*100:.1f} ± {np.std(separate_accuracy)*100:.1f}")
    score_substrings = [f"{i+1}={100.0*(np.sum(np.array(all_score_outputs)==i)/len(all_score_outputs)):.1f}%" for i in range(4)]
    print("Percentage classified as: " + ", ".join(score_substrings))