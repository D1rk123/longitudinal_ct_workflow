#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:18:36 2023

@author: des
"""

import numpy as np
import scipy
from matplotlib import pyplot as plt

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder


def parse_browning_scores(metadata_path):
    with open(metadata_path) as file:
        metadata = [line.rstrip().split(",") for line in file][1:]
    
    scores = {}
    for record in metadata:    
        apple_nr = int(record[0][1:])
        label = int(record[5])-1
        scores[apple_nr] = label
        
    return scores
    
def pad_to_square(img):
    size = max(img.shape)
    start = (np.array((size, size))-img.shape)//2
    
    padded_img = np.zeros((size, size), dtype=img.dtype)
    padded_img[start[0]:start[0]+img.shape[0], start[1]:start[1]+img.shape[1]] = img
    return padded_img

def plot_differences(recons, curr_days, reference_day, figures_folder, slice_num, apple_nr):  
    num_days = len(curr_days)
    fig, axs = plt.subplots(2, num_days, figsize=(5.7*num_days, 9))
    for i, day_tup in enumerate(curr_days):
        day, title = day_tup
        curr_slice = recons[day][slice_num,...]
        ref_slice = recons[reference_day][slice_num,...]
        print(curr_slice.shape)
        
        im1 = axs[0, i].imshow(pad_to_square(curr_slice), vmin=-0.1, vmax=1.1, cmap="Greys_r")
        plt.colorbar(im1, ax=axs[0, i])
        #axs[0, i].set_title(f"{title}")
        
        diff = pad_to_square(curr_slice-ref_slice)
        max_abs = np.max(np.abs(diff))
        quantile_abs = np.quantile(np.abs(diff), 0.999)
        print(f"min diff = {np.min(diff)}, max diff = {np.max(diff)}, quantile = {quantile_abs}")
        
        im2 = axs[1, i].imshow(diff, vmin=-0.6, vmax=0.6, cmap="RdBu")
        plt.colorbar(im2, ax=axs[1, i])
        #axs[1, i].set_title(f"Difference map")
    plt.tight_layout()
    plt.savefig(figures_folder / f"apple{apple_nr}_slice{slice_num}")
    plt.close()

if __name__ == "__main__":
    base_path = get_data_folder()
    recons_path = base_path / "recons_bh_corr_registered_crop"
    dms_path = base_path / "recons_bh_corr_registered_crop_dm"
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    figures_folder = experiment_folder / "figures"
    figures_folder.mkdir()
    
    reference_day = "2022-11-02 day 1"
    #all_days = sorted([d.name for d in (base_path / "projections").iterdir()])
    #all_days.remove(reference_day)
    all_days = [
        ("2022-11-02 day 1", "before storage"),
        ("2022-11-28 CA storage 3 weeks", "3 weeks in storage"),
        ("2023-01-09 CA storage 10 weeks", "10 weeks in storage"),
        ("2023-01-30 CA storage 13 weeks out day 1", "13 weeks in storage"),
        ("2023-03-06 CA storage 18 weeks out day 1", "18 weeks in storage"),
        ("2023-03-08 CA storage 18 weeks out day 3", "2 days out of storage"),
        ("2023-03-10 CA storage 18 weeks out day 5", "4 days out of storage"),
        ("2023-03-13 CA storage 18 weeks out day 8", "7 days out of storage"),
        ("2023-03-20 CA storage 18 weeks out day 15", "14 days out of storage")
    ]
    
    for i in [21]:#range(1, 87):
        apple_recon_dir = recons_path / f"{i}"
        if not apple_recon_dir.exists():
            print(f"Skipping {i} (folder does not exist)")
            continue
        curr_days = [dt for dt in all_days if (apple_recon_dir / dt[0]).exists()]
        if not (apple_recon_dir / reference_day).exists():
            print(f"Skipping {i} ({reference_day} not available)")
            continue
        
        print(f"Starting on {i}")
        
        reference_recon = ceu.load_stack(recons_path / f"{i}" / reference_day)
        reference_dm = ceu.load_stack(dms_path / f"{i}" / reference_day)
        core_center = int(scipy.ndimage.center_of_mass(reference_dm==-100)[0])
        
        recons = {reference_day : reference_recon}
        
        for day, title in curr_days:
            recon = ceu.load_stack(recons_path / f"{i}" / day)
            recons[day] = recon
            
            
        plot_differences(recons, curr_days, reference_day, figures_folder, core_center, i)
            
            
            
        
        
