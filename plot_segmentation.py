import tifffile
import numpy as np
from matplotlib import pyplot as plt

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from IG_compare import dm_to_masks

if __name__ == "__main__":
    base_path = get_data_folder()
    slice_path = base_path / "recons_bh_corr_registered_crop" / "70" / "2023-03-20 CA storage 18 weeks out day 15" / "output00295.tif"
    dm_path = base_path / "recons_bh_corr_registered_crop_dm" / "70" / "2023-03-06 CA storage 18 weeks out day 1" / "output00295.tif"
    inner_apple_dist = 12
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
    recon = tifffile.imread(slice_path)
    dm = tifffile.imread(dm_path)
    
    mask_foreground, mask_background, mask_outer_apple, mask_inner_apple, mask_core = dm_to_masks(dm, inner_apple_dist)

    color_mask = np.zeros((dm.shape[0], dm.shape[1], 3))
    color_mask += np.array((1, 0, 0), dtype=np.uint8)[None, None, :] * mask_outer_apple[:,:,None]
    color_mask += np.array((0, 1, 0), dtype=np.uint8)[None, None, :] * mask_inner_apple[:,:,None]
    color_mask += np.array((0, 0, 1), dtype=np.uint8)[None, None, :] * mask_core[:,:,None]
    
    combined = color_mask * 0.3 + (np.clip(recon, 0, 1)*0.7)[:,:,None]
    
    plt.figure(figsize=(4.2, 4.2))
    plt.imshow(combined)
    plt.plot((100), (100), c='#FF6666', label='outer apple', linewidth=7.0)
    plt.plot((100), (100), c='#66FF66', label='inner apple', linewidth=7.0)
    plt.plot((100), (100), c='#6666FF', label='core', linewidth=7.0)
    plt.legend()
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
    plt.tight_layout()
    plt.savefig(experiment_folder / "segmentation.png", dpi=300)
    plt.close()
