#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:39:48 2023

@author: des
"""
import numpy as np
from pathlib import Path

from skimage.filters import threshold_otsu
from skimage.measure import label

import ts_algorithms as tsa
import ct_experiment_utils as ceu

from folder_locations import get_results_folder
from fit_bh_params_juice import load_and_process_proj, make_geometries
from fix_scan_settings import fix_scan_settings

def reconstruct(scan_path):
    vg, pg, A = make_geometries(scan_path, "cwi-flexray-2022-10-28")
    y = load_and_process_proj(scan_path)
    y.nan_to_num_(1, 1)
    
    # apply beam hardening correction
    y_bh = 0.8372282981872559 * y + 0.13975094258785248 * (y**2) + 0.00014380142965819687 * (y**3) + -0.00028054710128344595
    
    # reconstruct
    recon = tsa.fdk(A, y_bh)
    
    # apply porosity mapping
    background_mean = -0.00015140167670324445
    juice_mean = 0.02493796870112419
    recon = (recon-background_mean)/(juice_mean-background_mean)
    
    return recon

def calc_bounding_box(recon):
    #use otsu thresholding to segment the apple
    threshold = threshold_otsu(recon.numpy())
    mask = recon.numpy() > threshold
    
    # find the largest connected component
    # https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
    labels = label(mask)
    assert( labels.max() != 0 ) # assume at least 1 CC
    mask = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
    coords = np.stack(np.nonzero(mask))
    min_vec = np.min(coords, axis=1)
    max_vec = np.max(coords, axis=1)
    return min_vec, max_vec
    
def crop(recon, min_vec, max_vec, crop_vec):
    size_vec = np.array(list(recon.size()))
    center_vec = (max_vec - min_vec).astype(np.float64)/2+min_vec
    start_vec = center_vec - crop_vec.astype(np.float64)/2
    start_vec = np.clip(start_vec, 0, size_vec-crop_vec)
    start_vec = np.floor(start_vec).astype(int)
    stop_vec = start_vec+crop_vec
    return recon[start_vec[0]:stop_vec[0], start_vec[1]:stop_vec[1], start_vec[2]:stop_vec[2]]

if __name__ == "__main__":
    base_path = Path("/export/scratch2/des/scans_breaburn/")
    proj_path = base_path / "projections"
    recons_path = base_path / "recons_bh_corr_crop"
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    results = experiment_folder / "recon_bboxes.csv"
    
    days = sorted([dir.name for dir in proj_path.iterdir() if dir.is_dir()])
    
    crop_vec = np.array((720, 690, 690), dtype=int)

    
    with open(results, "w") as file:
        file.write("day,apple,min_vec[0],min_vec[1],min_vec[2],max_vec[0],max_vec[1],max_vec[2]\n")
    for day in days:
        day_path = proj_path / day
        apples = sorted(
            [dir.name for dir in day_path.iterdir() if dir.is_dir()],
            key = lambda a : int(a)
            )
        (recons_path / day).mkdir(parents=True)
        for apple in apples:
            try:
                scan_path = proj_path / day / apple
                save_path = recons_path / day / (apple + "_recon")

                fix_scan_settings(scan_path / "scan settings.txt", fix_mag_det_phys=True, fix_cancelled=True)
                recon = reconstruct(scan_path)
                min_vec, max_vec = calc_bounding_box(recon)
                recon = crop(recon, min_vec, max_vec, crop_vec)
                ceu.save_stack(save_path, recon)
                
                result_str = f"{day},{apple},{min_vec[0]},{min_vec[1]},{min_vec[2]},{max_vec[0]},{max_vec[1]},{max_vec[2]}\n"
            except Exception as e:
                result_str = f"{day},{apple},{repr(e)},,,,,\n"
            print(result_str)
            with open(results, "a") as file:
                file.write(result_str)

