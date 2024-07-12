#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:18:36 2023

@author: des
"""

import numpy as np
import scipy
from matplotlib import pyplot as plt
import tomosipo as ts

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

def make_projection_operator(volume, voxel_size, src_obj_dist, src_det_dist):
    pg = ts.cone(
        shape = (1200, 1200),
        size = (1200*voxel_size, 1200*voxel_size),
        src_orig_dist = src_obj_dist,
        src_det_dist = src_det_dist
        )
    vg = ts.volume(
        shape = volume.shape,
        size = (volume.shape[0]*voxel_size, volume.shape[1]*voxel_size, volume.shape[2]*voxel_size)
    )
    return ts.operator(vg, pg)
        
        
    

def plot_projected_differences(projs, curr_days, reference_day, figures_folder, apple_nr):  
    num_days = len(curr_days)
    fig, axs = plt.subplots(2, num_days, figsize=(5.7*num_days, 9))
    
    for i, day_tup in enumerate(curr_days):
        day, title = day_tup
        
        print(f"proj max = {np.max(projs[day])}, proj.shape = {projs[day].shape}")
        im1 = axs[0, i].imshow(projs[day], vmin=-0.1, vmax=60.1, cmap="Greys_r")
        plt.colorbar(im1, ax=axs[0, i])
        #axs[0, i].set_title(f"{title}")
        
        diff = projs[day]-projs[reference_day]
        max_abs = np.max(np.abs(diff))
        quantile_abs = np.quantile(np.abs(diff), 0.999)
        print(f"min diff = {np.min(diff)}, max diff = {np.max(diff)}, quantile = {quantile_abs}")
        
        im2 = axs[1, i].imshow(diff, vmin=-6, vmax=6, cmap="RdBu") # 
        plt.colorbar(im2, ax=axs[1, i])
        #axs[1, i].set_title("Difference map")
    plt.tight_layout()
    plt.savefig(figures_folder / f"apple{apple_nr}")
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
        
        reference_dm = ceu.load_stack(dms_path / f"{i}" / reference_day)
        mask = reference_dm < -0.5
        reference_recon = ceu.load_stack(recons_path / f"{i}" / reference_day) * mask
        
        recons = {reference_day : reference_recon}
        
        for day, title in curr_days:
            recon = ceu.load_stack(recons_path / f"{i}" / day, stack_axis=0)
            recons[day] = recon * mask
            
        A = make_projection_operator(reference_recon, 0.13, 559.949001, 1099.363734)
        projs = {}
        for day, title in curr_days:
            projs[day] = np.flipud(A(recons[day])[:, 0, :])
            
            
        plot_projected_differences(projs, curr_days, reference_day, figures_folder, i)
            
            
            
        
        
