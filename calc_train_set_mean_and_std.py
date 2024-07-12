import numpy as np

import ct_experiment_utils as ceu
from folder_locations import get_data_folder
from nn_detect_browning import parse_metadata, parse_core_slice_nrs

if __name__ == "__main__":
    base_path = get_data_folder()
    recons_path = base_path / "recons_bh_corr_registered_crop"
    dms_path = base_path / "recons_bh_corr_registered_crop_dm"
    metadata_path = base_path / "Brae_browning_score.csv"
    centers_of_mass_path = base_path / "centers_of_mass.csv"
    
    selected_days = [
        "2023-01-23 CA storage 10 weeks out 2 weeks",
        "2023-01-30 CA storage 10 weeks out 3 weeks",
        "2023-02-13 CA storage 13 weeks out day 15",
        "2023-03-13 CA storage 18 weeks out day 8",
        "2023-03-20 CA storage 18 weeks out day 15"
    ]
    
    train_set_nrs = [41, 42, 50, 49, 16, 19, 24, 28, 33, 45, 51, 53, 54, 55, 56,
                     57, 58, 59, 60, 61, 63, 6, 29, 64, 66, 68, 72, 74, 77, 80,
                     81, 82, 85, 22, 34, 35, 46, 67, 70, 71, 73, 75, 76, 78, 79]
    
    region_size = 0.2
    
    core_slice_nrs = parse_core_slice_nrs(centers_of_mass_path)
    scans = parse_metadata(metadata_path, recons_path, core_slice_nrs, selected_days, train_set_nrs)
    
    means = []
    stds = []
    for scan in scans:
        print(f"  ===  Scan {scan.apple_nr}  ===")
        min_slice_nr = round(scan.center_slice-(region_size/2*scan.num_slices))
        max_slice_nr = round(scan.center_slice+(region_size/2*scan.num_slices))
        
        dm_path = next((dms_path / str(scan.apple_nr)).iterdir())
        dm = ceu.load_stack(dm_path, range_start=min_slice_nr, range_stop=max_slice_nr)
        mask = np.logical_not(dm < 0)

        for recon_path in (recons_path / str(scan.apple_nr)).iterdir():
            recon = ceu.load_stack(recon_path, range_start=min_slice_nr, range_stop=max_slice_nr)
            masked_recon = np.ma.MaskedArray(recon, mask)
            
            means.append(np.ma.mean(masked_recon))
            stds.append(np.ma.std(masked_recon))
            print(f"Mean = {means[-1]}, std = {stds[-1]}")
    
    print("  ===  Finished  ===  ")
    print(f"Mean = {np.mean(means)}, std = {np.mean(stds)}")
