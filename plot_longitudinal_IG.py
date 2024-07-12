#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:51:45 2024

@author: des
"""
import sys
import torch
import tifffile
import numpy as np
import albumentations as A
import albumentations.pytorch as Ap
from matplotlib import pyplot as plt
import cv2
import time
from tqdm import tqdm

from captum.attr import IntegratedGradients

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from nn_detect_browning import parse_metadata, LRegressionModule, load_slice_centered, parse_core_slice_nrs
from invertible_augmentations import generate_no_augmentations, generate_flip_rot90_augmentations, generate_random_CT_augmentations

def dm_to_masks(dm, inner_apple_dist):
    mask_outer_apple = np.logical_and(dm<0, dm>=-inner_apple_dist)
    mask_inner_apple = np.logical_and(dm<-inner_apple_dist, dm>-100)
    mask_core = dm==-100
    mask_background = dm>=0
    mask_foreground = torch.from_numpy(dm<0)[None, ...]
    
    return mask_foreground, mask_background, mask_outer_apple, mask_inner_apple, mask_core   

class SmoothIG:
    def __init__(self, scans, augmentations, inner_apple_dist, num_imgs, region_size, experiment_folder, checkpoint_path, recons_path, masks_path):
        self.scans = scans
        self.augmentations = augmentations
        self.inner_apple_dist = inner_apple_dist
        self.num_imgs = num_imgs
        self.region_size = region_size
        
        self.experiment_folder = experiment_folder
        self.recons_path = recons_path
        self.masks_path = masks_path
        
        self.l_module_gc = LRegressionModule.load_from_checkpoint(checkpoint_path)
        self.l_module_gc.eval()
        self.l_module = LRegressionModule.load_from_checkpoint(checkpoint_path)
        self.l_module.freeze()
        self.l_module.eval()
        
        self.ig = IntegratedGradients(self.l_module_gc, multiply_by_inputs=True)
        
    def run_sub_experiment(self, name, ig_augmentations):
        sub_folder = self.experiment_folder / name
        sub_folder.mkdir()
        figures_folder = sub_folder / "figures"
        figures_folder.mkdir()
        tiffs_folder = sub_folder / "tiffs"
        tiffs_folder.mkdir()
        
        file = open(sub_folder / "results.csv", "w")
        file.write("apple_nr,slice_nr,ground_truth,rounded_median_result,mean_result,mean_baseline_result,ig_fraction_outer_apple,ig_fraction_inner_apple,"
            "ig_fraction_core,ig_fraction_background,diff_fraction_outer_apple,"
            "diff_fraction_inner_apple,diff_fraction_core,diff_fraction_background\n")
        
        for scan in self.scans:
            if not scan.apple_nr == 27:
                continue
            scan_tiffs_folder = tiffs_folder / f"{scan.apple_nr}"
            scan_tiffs_folder.mkdir()
            scan_figures_folder = figures_folder  / f"{scan.apple_nr}"
            scan_figures_folder.mkdir()

            min_slice_nr = round(scan.center_slice-(self.region_size/2*scan.num_slices))
            max_slice_nr = round(scan.center_slice+(self.region_size/2*scan.num_slices))
            results = []
            baseline_results = []
            
            for slice_nr in np.round(np.linspace(min_slice_nr, max_slice_nr, self.num_imgs)).astype(int):
                if not slice_nr == 315:
                    continue
                recon_slice_path = scan.path / f"output{slice_nr:05d}.tif"
                mask_slice_path = next((self.masks_path / str(scan.apple_nr)).iterdir()) / recon_slice_path.name
                recon_slice_path_day0 = self.recons_path / recon_slice_path.parents[1].name / day0 / recon_slice_path.name
                if not recon_slice_path.exists() or not recon_slice_path_day0.exists():
                    continue
                    
                img = load_slice_centered((690, 690), recon_slice_path, mask_slice_path, 0)
                img_day0 = load_slice_centered((690, 690), recon_slice_path_day0, mask_slice_path, 0)
                
                img_diff = (img-img_day0)
                
                dm = load_slice_centered((690, 690), mask_slice_path)
                mask_foreground, mask_background, mask_outer_apple, mask_inner_apple, mask_core = dm_to_masks(dm, self.inner_apple_dist)

                heatmaps = []
                results = []
                for ig_augmentation in tqdm(ig_augmentations):
                    img_aug = self.augmentations(image=ig_augmentation.apply(img))["image"]
                    baseline_aug = self.augmentations(image=ig_augmentation.apply(img_day0))["image"]
                    
                    results.append(self.l_module(img_aug[None,:,:,:].cuda())[0, 0].cpu().numpy())
                    baseline_results.append(self.l_module(baseline_aug[None,:,:,:].cuda())[0, 0].cpu())
                    
                    heatmap = self.ig.attribute(
                        img_aug[None, ...].cuda(),
                        target=None,
                        baselines=baseline_aug[None, ...].cuda(),
                        internal_batch_size=3,
                        n_steps=100).detach().cpu()[0,0].numpy()
                    
                    heatmap = ig_augmentation.apply_inverse(heatmap)
                    heatmaps.append(heatmap)
                    
                heatmap = np.mean(np.stack(heatmaps), axis=0)
                result = np.clip(np.round(np.median(results)), 0, 3)
                mean_result = np.mean(results)
                mean_baseline_result = np.mean(baseline_results)
            
                abs_heatmap = np.abs(heatmap)
                abs_diff_img = np.abs(img_diff)
                sum_heatmap = np.sum(abs_heatmap)
                sum_img_diff = np.sum(abs_diff_img)
                
                ig_fraction_outer_apple = np.sum(abs_heatmap*mask_outer_apple)/sum_heatmap
                ig_fraction_inner_apple = np.sum(abs_heatmap*mask_inner_apple)/sum_heatmap
                ig_fraction_core        = np.sum(abs_heatmap*mask_core)/sum_heatmap
                ig_fraction_background  = np.sum(abs_heatmap*mask_background)/sum_heatmap
                
                diff_fraction_outer_apple = np.sum(abs_diff_img*mask_outer_apple)/sum_img_diff
                diff_fraction_inner_apple = np.sum(abs_diff_img*mask_inner_apple)/sum_img_diff
                diff_fraction_core        = np.sum(abs_diff_img*mask_core)/sum_img_diff
                diff_fraction_background  = np.sum(abs_diff_img*mask_background)/sum_img_diff
                
                line = f"{scan.apple_nr},{slice_nr},{scan.label},{result},{mean_result},{mean_baseline_result},{ig_fraction_outer_apple},{ig_fraction_inner_apple},{ig_fraction_core},{ig_fraction_background},{diff_fraction_outer_apple},{diff_fraction_inner_apple},{diff_fraction_core},{diff_fraction_background}"
                print(line)
                file.write(line+"\n")
                
                tifffile.imwrite(str(scan_tiffs_folder / f"{slice_nr}.tiff"), heatmap)
                
                plt.figure(figsize=(13, 3.5))
                plt.subplot(131)
                plt.imshow(img, vmin=0, vmax=1.1, cmap="Greys_r")
                plt.title("CT slice")
                plt.colorbar(label="Juice equivalent density")
                plt.xticks(np.arange(0, 690, 100))
                plt.yticks(np.arange(0, 690, 100))
                plt.grid()
                plt.subplot(132)
                plt.title("Difference image")
                plt.imshow(img_diff, vmin=-0.6, vmax=0.6, cmap="RdBu")
                plt.colorbar(label="Difference from before storage")
                plt.xticks(np.arange(0, 690, 100))
                plt.yticks(np.arange(0, 690, 100))
                plt.grid()
                plt.subplot(133)
                plt.imshow(heatmap, vmin=-0.0015, vmax=0.0015, cmap="PiYG")
                plt.colorbar(label="Attribution to label")
                plt.title("Integrated gradients heatmap")
                plt.xticks(np.arange(0, 690, 100))
                plt.yticks(np.arange(0, 690, 100))
                plt.grid()
                plt.tight_layout()
                plt.savefig(scan_figures_folder / f"{scan.apple_nr}_{slice_nr}.png", dpi=300)
                plt.close()
        file.close()

if __name__ == "__main__":
    np.random.seed(int(time.time())) 
    base_path = get_data_folder()
    recons_path = base_path / "recons_bh_corr_registered_crop"
    masks_path = base_path / "recons_bh_corr_registered_crop_dm"
    metadata_path = base_path / "Brae_browning_score.csv"
    centers_of_mass_path = base_path / "centers_of_mass.csv"
    
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
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
        
    day0 = "2022-11-02 day 1"
    inner_apple_dist = 12
    num_imgs = 10
    
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
    
    ig_runner = SmoothIG(scans, augmentations, inner_apple_dist, num_imgs, region_size, experiment_folder, checkpoint_path, recons_path, masks_path)
    ig_runner.run_sub_experiment("no_augmentations", generate_no_augmentations())
    ig_runner.run_sub_experiment("flip_rotate_augmentations", generate_flip_rot90_augmentations([690, 690]))
    ig_runner.run_sub_experiment("random100_augmentations", generate_random_CT_augmentations(100, [690, 690]))
   
    
    
