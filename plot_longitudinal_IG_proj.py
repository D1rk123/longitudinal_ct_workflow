#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:51:45 2024

@author: des
"""
import sys
import torchvision
import tifffile
import numpy as np
import albumentations as A
import albumentations.pytorch as Ap
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm

from captum.attr import IntegratedGradients

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from nn_detect_browning_proj_optuna import LRegressionModule, parse_metadata_proj
from invertible_augmentations import generate_no_augmentations, generate_flip_augmentations, generate_random_proj_augmentations

class SmoothIG_proj:
    def __init__(self, pstacks, augmentations, num_imgs, experiment_folder, checkpoint_path, baselines_path):
        self.pstacks = pstacks
        self.augmentations = augmentations
        self.num_imgs = num_imgs
        
        self.experiment_folder = experiment_folder
        self.checkpoint_path = checkpoint_path
        self.baselines_path = baselines_path
        
        self.l_module_gc = LRegressionModule.load_from_checkpoint(checkpoint_path)
        self.l_module_gc.eval()
        self.l_module = LRegressionModule.load_from_checkpoint(checkpoint_path)
        self.l_module.freeze()
        self.l_module.eval()
        
        self.crop_top = self.l_module.trial.params["crop_top"]
        self.crop_bottom = self.l_module.trial.params["crop_bottom"]
        
        self.ig = IntegratedGradients(self.l_module_gc, multiply_by_inputs=True)
        
    def run_sub_experiment(self, name, ig_augmentations):
        sub_folder = self.experiment_folder / name
        sub_folder.mkdir()
        figures_folder = sub_folder / "figures"
        figures_folder.mkdir()
        tiffs_folder = sub_folder / "tiffs"
        tiffs_folder.mkdir()
        
        file = open(sub_folder / "results.csv", "w")
        file.write("apple_nr,proj_nr,ground_truth,rounded_median_result,mean_result,mean_baseline_result\n")
        print("starting")
        
        for pstack in self.pstacks:
            print(f"test {pstack.apple_nr}")
            if not pstack.apple_nr == 27:
                print(f"skipping apple {pstack.apple_nr}")
                continue
            scan_tiffs_folder = tiffs_folder / f"{pstack.apple_nr}"
            scan_tiffs_folder.mkdir()
            scan_figures_folder = figures_folder  / f"{pstack.apple_nr}"
            scan_figures_folder.mkdir()

            results = []
            
            projs = ceu.load_stack(pstack.path, prefix="output")
            baseline_path = baselines_path / pstack.path.name
            projs_bl = ceu.load_stack(baseline_path, prefix="output")
            for i in np.round(np.linspace(0, projs_bl.shape[0], self.num_imgs, endpoint=False)).astype(int):
                if not i == 336:
                    continue
                img = projs[i]
                img_bl = projs_bl[i] 
                img_diff = (img-img_bl)

                heatmaps = []
                results = []
                baseline_results = []
                for ig_augmentation in tqdm(ig_augmentations):
                    img_aug = self.augmentations(image=ig_augmentation.apply(img))["image"]
                    baseline_aug = self.augmentations(image=ig_augmentation.apply(img_bl))["image"]
                    
                    results.append(self.l_module(img_aug[None,:,:,:].cuda())[0, 0].cpu())
                    baseline_results.append(self.l_module(baseline_aug[None,:,:,:].cuda())[0, 0].cpu())
                    heatmap = self.ig.attribute(
                        img_aug[None, ...].cuda(),
                        target=None,
                        baselines=baseline_aug[None, ...].cuda(),
                        internal_batch_size=3,
                        n_steps=100).detach().cpu()[0,0].numpy()
                        
                    heatmap = np.pad(heatmap, ((self.crop_top, self.crop_bottom), (0, 30)))
                    a = results[-1]-baseline_results[-1]
                    b = np.sum(heatmap)
                    print(f"result-baseline_result = {a}, heatmap sum = {b}, difference = {a-b}")
                    
                    heatmap = ig_augmentation.apply_inverse(heatmap)
                    heatmaps.append(heatmap)
                    
                heatmap = np.mean(np.stack(heatmaps), axis=0)
                result = np.clip(np.round(np.median(results)), 0, 3)
                mean_result = np.mean(results)
                mean_baseline_result = np.mean(baseline_results)
                
                line = f"{pstack.apple_nr},{i},{pstack.label},{result},{mean_result},{mean_baseline_result}"
                print(line)
                file.write(line+"\n")

                tifffile.imwrite(str(scan_tiffs_folder / f"{i}.tiff"), heatmap)
                
                plt.figure(figsize=(13, 3.5))
                plt.subplot(131)
                plt.imshow(img, vmin=0, vmax=1.5, cmap="Greys_r")
                #plt.title(f"Label = {pstack.label}, NN prediction = {result}")
                plt.title("Radiograph")
                plt.colorbar(label="Attenuation")
                plt.xticks(np.arange(0, 720, 100))
                plt.yticks(np.arange(0, 690, 100))
                plt.grid()
                plt.subplot(132)
                plt.title("Difference image")
                plt.imshow(img_diff, vmin=-0.3, vmax=0.3, cmap="RdBu") # 
                plt.colorbar(label="Difference from before storage")
                plt.xticks(np.arange(0, 720, 100))
                plt.yticks(np.arange(0, 690, 100))
                plt.grid()
                plt.subplot(133)
                plt.imshow(heatmap, vmin=-0.0015, vmax=0.0015, cmap="PiYG")
                plt.colorbar(label="Attribution to label")
                plt.title("Integrated gradients heatmap")
                plt.xticks(np.arange(0, 720, 100))#, labels=[str(i) if i % 100 == 0 else "" for i in np.arange(0, 690, 50)])
                plt.yticks(np.arange(0, 690, 100))
                plt.grid()
                plt.tight_layout()
                plt.savefig(scan_figures_folder / f"{pstack.apple_nr}_{i}.png", dpi=300)
                plt.close()
        file.close()

if __name__ == "__main__":
    base_path = get_data_folder()
    ff_proj_path = base_path / "projections_flat_crop"
    metadata_path = base_path / "Brae_browning_score.csv"
    experiments_path = get_results_folder()
    
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
    if sys.argv[1] == "d":
        checkpoint_path = experiments_path / "2024-04-01_nn_detect_browning_proj_optuna_1_detection_final" / "trial_0" / "checkpoints" / "mae_val" / "best_mae_val_epoch=7387_mae_val=0.2954833508.ckpt"
        baselines_path = experiments_path / "2024-04-15_make_registered_day0_projs_5_detection_final"
        
        selected_days = [
            "2023-01-23 CA storage 10 weeks out 2 weeks",
            "2023-01-30 CA storage 10 weeks out 3 weeks",
            "2023-02-13 CA storage 13 weeks out day 15",
            "2023-03-13 CA storage 18 weeks out day 8",
            "2023-03-20 CA storage 18 weeks out day 15"
        ]
        day_dict = None
        
    elif sys.argv[1] == "p":
        checkpoint_path = experiments_path / "2024-04-03_nn_detect_browning_proj_optuna_1_prediction_final" / "trial_0" / "checkpoints" / "mae_val" / "best_mae_val_epoch=490_mae_val=0.8732241988.ckpt"
        baselines_path = experiments_path / "2024-04-15_make_registered_day0_projs_6_prediction_final"
        
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
    num_imgs = 10
    
    available_test_set_nrs = [27]#[13, 17, 8, 18, 4, 25, 36, 37, 32, 23, 38, 14, 3, 27]
    
    # Calculated over all projections over all days of the apples of the training set
    dataset_mean = np.float32(0.54003301858902).item()
    dataset_std = np.float32(0.573314089958484).item()
    
    l_module = LRegressionModule.load_from_checkpoint(checkpoint_path)
    crop_top = l_module.trial.params["crop_top"]
    crop_bottom = l_module.trial.params["crop_bottom"]
    del l_module

    pstacks = parse_metadata_proj(metadata_path, ff_proj_path, selected_days, available_test_set_nrs, day_dict)
    augmentations = A.Compose([
        A.Crop(x_min=0, x_max=690, y_min=crop_top, y_max=690-crop_bottom),
        A.CropAndPad(px=-1, keep_size=False),
        A.CropAndPad(px=1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
        A.Normalize(mean=dataset_mean, std=dataset_std, max_pixel_value=1.0),
        Ap.ToTensorV2()
        ])
        
    ig_runner = SmoothIG_proj(pstacks, augmentations, num_imgs, experiment_folder, checkpoint_path, baselines_path)
    ig_runner.run_sub_experiment("no_augmentations", generate_no_augmentations())
    ig_runner.run_sub_experiment("flip_augmentations", generate_flip_augmentations([690, 720]))
    ig_runner.run_sub_experiment("random100_augmentations", generate_random_proj_augmentations(100, [690, 720]))
   
    
    
