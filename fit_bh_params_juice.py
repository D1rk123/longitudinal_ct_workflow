#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:50:59 2023

@author: des
"""
import numpy as np
import torch
from pathlib import Path
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_dilation, disk
import tifffile
from tqdm import tqdm

import tomosipo as ts
import ts_algorithms as tsa
import ts_flexray as tsf
import ct_experiment_utils as ceu

from folder_locations import get_results_folder
from autograd_operator import AutogradOperator

class MomentumOptimizer():
    def __init__(self, param_init_values, learning_rates, momentums):
        self.params = param_init_values
        self.dtype = param_init_values[0].dtype
        self.device = param_init_values[0].device
        self.steps = [torch.zeros_like(param, requires_grad=False) for param in self.params]
        self.learning_rates = self._prepare_list(learning_rates)
        self.momentums = self._prepare_list(momentums)
        
    def _prepare_list(self, li):
        result = []
        for el in li:
            if not isinstance(el, torch.Tensor):
                result.append(torch.tensor(el, dtype=self.dtype, device=self.device))
            else:
                result.append(el.to(dtype=self.dtype, device=self.device))
        return result
        
    def step(self):
        with torch.no_grad():
            for param, step, learning_rate, momentum in zip(self.params, self.steps, self.learning_rates, self.momentums):
                step[...] = param.grad * learning_rate + momentum * step
                param -= step
                param.grad.zero_()

# Preprocesses the projection data without making copies
def preprocess_in_place(y, dark, flat):
    dark = dark[:, None, :]
    flat = flat[:, None, :]
    y -= dark
    y /= (flat - dark)
    torch.log_(y)
    y *= -1

def make_geometries(proj_path, profile):
    vg, pg, geom_dict = tsf.make_flexray_geometries(proj_path / "scan settings.txt", profile)
    A = ts.operator(vg, pg)
    return vg, pg, A

# Loads a tiff file and converts it to a float32 torch tensor
def load_tiff_to_torch(path):
    return torch.flip(torch.from_numpy(tifffile.imread(str(path)).astype(np.float32)), dims=[0])

def load_and_process_proj(proj_path):
    # Load the dark field and flat field images separately
    dark_field = load_tiff_to_torch(proj_path / "di000000.tif")
    flat_field = load_tiff_to_torch(proj_path / "io000000.tif")
    
    # Load the projection data and apply the log preprocessing
    y = torch.from_numpy(ceu.load_stack(proj_path, prefix="scan", dtype=np.float32, stack_axis=1, range_stop=-1))
    y = torch.flip(y, dims=[0])
    preprocess_in_place(y, dark_field, flat_field)
    return y

def make_masks(recon_no_bh):
    recon = torch.clone(recon_no_bh).numpy()
    
    recon[0:75, ...] = 0
    recon[500:, ...] = 0
    
    thresh = threshold_otsu(recon)
    
    otsu_mask = recon > thresh
    inner_mask = binary_erosion(otsu_mask, disk(4)[None, :, :])
    
    outer_mask = np.zeros_like(otsu_mask)
    outer_mask[75:500, (956//2)-350:(956//2)+351, (956//2)-350:(956//2)+351] = disk(350)[None, ...]
    outer_mask = outer_mask ^ binary_dilation(otsu_mask, disk(2)[None, :, :])
    
    return torch.from_numpy(inner_mask), torch.from_numpy(outer_mask)
    

if __name__ == "__main__":
    proj_path = Path("/export/scratch2/des/juice_sample/2023-03-27 juice sample 5avg")
    
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder(), name="lin_bh_corr_fit_juice")
    inner_mask_path = experiment_folder / "inner_mask"
    outer_mask_path = experiment_folder / "outer_mask"
    recon_corr_path = experiment_folder / "bh_recon"
    params_path = experiment_folder / "params.txt"
    
    vg, pg, A = make_geometries(proj_path, "cwi-flexray-2022-10-28")
    y = load_and_process_proj(proj_path)
    
    recon_no_bh = tsa.fdk(A, y)
    inner_mask, outer_mask = make_masks(recon_no_bh)
    
    ceu.save_stack(inner_mask_path, inner_mask, exist_ok=True)
    ceu.save_stack(outer_mask_path, outer_mask, exist_ok=True)
    print("Finished all preprocessing")
    print("Now starting the beam hardening parameter optimization")
    
    A_ag = AutogradOperator(A)
    
    #y = y.cuda()
    bh_params = torch.tensor((0, 1, 0, 0), dtype=torch.float32)
    bh_params.requires_grad_(True)
    
    inner_sum = torch.sum(inner_mask)
    outer_sum = torch.sum(outer_mask)
    
    inner_mean = torch.sum(recon_no_bh*inner_mask)/inner_sum
    outer_mean = torch.sum(recon_no_bh*outer_mask)/outer_sum
    
    optimizer = MomentumOptimizer([bh_params], [torch.tensor([50, 50, 50, 50], dtype=torch.float32)], [torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)])
    num_iterations = 5
    
    for i in tqdm(range(num_iterations)):
        y_bh = bh_params[1] * y + bh_params[2] * (y**2) + bh_params[3] * (y**3) + bh_params[0]

        recon = tsa.fdk(A=A_ag, y=y_bh)

        inner_mse = torch.sum(((recon-inner_mean)*inner_mask)**2)/inner_sum
        outer_mse = torch.sum(((recon-outer_mean)*outer_mask)**2)/outer_sum
        error = (inner_mse + outer_mse) / 2
        print(f"error = {error}")
        
        error.backward()
        print(f"bh_params = {bh_params}")
        print(f"bh_params grad = {bh_params.grad}")
        optimizer.step()
    

    ceu.save_stack(recon_corr_path, recon.detach(), exist_ok=True)
    recon = tsa.fdk(A=A_ag, y=y_bh)
    inner_mean_corr = torch.sum(recon.detach().cpu()*inner_mask)/inner_sum
    outer_mean_corr = torch.sum(recon.detach().cpu()*outer_mask)/outer_sum
    with open(params_path, "w") as file:
        file.write(f"y_bh = {bh_params[1].detach().numpy()} * y + {bh_params[2].detach().numpy()} * (y**2) + {bh_params[3].detach().numpy()} * (y**3) + {bh_params[0].detach().numpy()}\n")
        file.write(f"bh corrected inner mean = {inner_mean_corr}\n")
        file.write(f"bh corrected outer mean = {outer_mean_corr}\n")
