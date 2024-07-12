#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:39:48 2023

@author: des
"""
import numpy as np
import torch
from pathlib import Path
import tifffile

from skimage.measure import centroid
import ct_experiment_utils as ceu

from folder_locations import get_results_folder

# Preprocesses the projection data without making copies
def preprocess_in_place(y, dark, flat):
    dark = dark[:, None, :]
    flat = flat[:, None, :]
    y -= dark
    y /= (flat - dark)
    torch.log_(y)
    y *= -1

# Loads a tiff file and converts it to a float32 torch tensor
def load_tiff_to_torch(path):
    return torch.from_numpy(tifffile.imread(str(path)).astype(np.float32))

def load_and_process_proj(proj_path, range_step=1):
    # Load the dark field and flat field images separately
    dark_field = load_tiff_to_torch(proj_path / "di000000.tif")
    flat_field = load_tiff_to_torch(proj_path / "io000000.tif")
    
    # Load the projection data and apply the log preprocessing
    y = torch.from_numpy(ceu.load_stack(proj_path, prefix="scan", dtype=np.float32, stack_axis=1, range_stop=-1, range_step=range_step))
    preprocess_in_place(y, dark_field, flat_field)
    return y
    
def test_edges(img):
    left_max = torch.max(img[:, 0])
    right_max = torch.max(img[:, -1])
    top_max = torch.max(img[0, :])
    bottom_max = torch.max(img[-1, :])
    if left_max > 0.2:
        print("High left edge")
    if right_max > 0.2:
        print("High right edge")
    if top_max > 0.2:
        print("High top edge")
    if bottom_max > 0.2:
        print("High bottom edge")

def crop_stack(stack, crop_vec):
    crop_result = torch.zeros((crop_vec[0], stack.size()[1], crop_vec[1]), dtype=torch.float32)
    size_vec = np.array(list(stack.size())).take((0, 2))
    for i in range(stack.size()[1]):
        center_vec = centroid(stack[:, i, :].numpy())
        start_vec = center_vec - crop_vec.astype(np.float64)/2
        start_vec = np.clip(start_vec, 0, size_vec-crop_vec)
        start_vec = np.floor(start_vec).astype(int)
        stop_vec = start_vec+crop_vec
        crop_result[:, i, :] = stack[start_vec[0]:stop_vec[0], i, start_vec[1]:stop_vec[1]]
        test_edges(crop_result[:, i, :])
    return crop_result

if __name__ == "__main__":
    proj_path = Path("/export/scratch2/des/scans_breaburn/projections")
    flatfielded_path = Path("/export/scratch3/des/scans_breaburn/projections_flat_crop")
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
    days = sorted([dir.name for dir in proj_path.iterdir() if dir.is_dir()])
    #tuples = (("2023-03-20 CA storage 18 weeks out day 15", "78"), ("2022-11-02 day1", "3"), ("2022-11-02 day1", "22"))
    #tuples = (("2023-01-30 CA storage 13 weeks out day1","5"), ("2022-11-02 day1","5"), ("2022-11-02 day1", "7"))
    
    crop_vec = np.array((690, 720), dtype=int)

    for day in days:
        day_path = proj_path / day
        apples = sorted(
            [dir.name for dir in day_path.iterdir() if dir.is_dir()],
            key = lambda a : int(a)
            )
        (flatfielded_path / day).mkdir(parents=True)
        for apple in apples:
            #if (day, apple) in tuples:
            #    print(f"Found {(day, apple)}")
            #else:
            #    continue
            scan_path = proj_path / day / apple
            save_path = flatfielded_path / day / (apple + "_ff")
            print(save_path)

            stack = load_and_process_proj(scan_path, 3)
            stack.nan_to_num_(1, 1)
            stack = crop_stack(stack, crop_vec)
            ceu.save_stack(save_path, stack, stack_axis=1)

