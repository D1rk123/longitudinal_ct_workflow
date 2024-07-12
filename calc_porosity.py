import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder

if __name__ == "__main__":
    base_path = get_data_folder()
    recons_path = base_path / "recons_bh_corr_registered_crop"
    dms_path = base_path / "recons_bh_corr_registered_crop_dm"
    
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())

    min_peel_dist = 1
    outer_core_dist = 12
    
    file = open(experiment_folder / "porosities.csv", "w")
    file.write("apple_nr,day_name,volume_apple,volume_outer_core,porosity_apple,porosity_outer_core\n")
    
    plot_data_full = []
    plot_data_inner_core = []
    for apple_nr in range(1, 87):
        apple_path = recons_path / f"{apple_nr}"
        if not apple_path.exists() or apple_nr == 30: # Apple 30 is rotten
            print(f"Skipping {apple_nr}")
            continue
        dm_path = next((dms_path / f"{apple_nr}").iterdir())
        recon_paths = sorted([p for p in apple_path.iterdir()])
        
        dm = ceu.load_stack(dm_path)
        mask_apple = dm<-min_peel_dist
        volume_apple = np.sum(mask_apple.astype(int))
        
        mask_outer_core = np.logical_and(dm<-outer_core_dist, dm!=-100)
        volume_outer_core = np.sum(mask_outer_core.astype(int))
        
        for recon_path in recon_paths:
            recon = np.clip(ceu.load_stack(recon_path), 0, 1)
            
            porosity_apple = np.sum((1-recon)*mask_apple)/volume_apple
            porosity_outer_core = np.sum((1-recon)*mask_outer_core)/volume_outer_core
            
            porosity = 0.5
            line = f"{apple_nr},{recon_path.name},{volume_apple},{volume_outer_core},{porosity_apple:.10f},{porosity_outer_core:.10f}"
            print(line)
            file.write(line+"\n")
            
