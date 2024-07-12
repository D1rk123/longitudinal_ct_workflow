import itertools
from pathlib import Path
import numpy as np
from skimage.measure import label
from skimage.segmentation import flood
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, ball
import scipy
from matplotlib import pyplot as plt

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder
from fix_scan_settings import extract_value

def segment_apple(image, threshold):
    mask = image > threshold
    
    # closing by 4 pixels
    # given the pixel size of ~0.130mm this roughly closes 8*0.13=1.04mm holes
    mask = binary_closing(mask, ball(4))
    
    # find the largest connected component
    # https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
    labels = label(mask)
    assert( labels.max() != 0 ) # assume at least 1 CC
    mask = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
    
    # fill all holes within the connected component
    mask = (flood(mask, (0,0,0)) == False)
    
    return mask
    
def vertical_fill(mask):
    result = np.zeros_like(mask)
    for i, j in itertools.product(range(mask.shape[1]), range(mask.shape[2])):
        nonzeros = np.nonzero(mask[:, i, j])[0]
        if len(nonzeros) != 0:
            result[nonzeros[0]:nonzeros[-1],i,j] = 1
        
    return result
    
def segment_core(recon, central_slice):
    region_offset0 = round(recon.shape[0]*0.20)
    region_offset1 = round(recon.shape[1]*0.25)
    region_offset2 = round(recon.shape[2]*0.25)
    central_region = recon[
        central_slice-region_offset0:central_slice+region_offset0,
        recon.shape[1]//2-region_offset1:recon.shape[1]//2+region_offset1,
        recon.shape[2]//2-region_offset2:recon.shape[2]//2+region_offset2]

    mask = central_region < 0.15
    mask = binary_dilation(mask, ball(5))
    mask = binary_dilation(mask, ball(5))

    # find the largest connected component
    # https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
    labels = label(mask)
    assert( labels.max() != 0 ) # assume at least 1 CC
    mask = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)

    mask = vertical_fill(mask)

    # Morphological closing (multiple smaller structuring elements are faster than one big one)
    mask = binary_dilation(mask, ball(5))
    mask = binary_dilation(mask, ball(5))
    mask = binary_dilation(mask, ball(5))
    mask = binary_erosion(mask, ball(5))
    mask = binary_erosion(mask, ball(5))
    mask = binary_erosion(mask, ball(5))

    # fill all holes within the connected component
    mask = (flood(mask, (0,0,0)) == False)
    
    result = np.zeros(recon.shape, dtype=bool)
    result[
        central_slice-region_offset0:central_slice+region_offset0,
        recon.shape[1]//2-region_offset1:recon.shape[1]//2+region_offset1,
        recon.shape[2]//2-region_offset2:recon.shape[2]//2+region_offset2
        ] = mask
        
    return result
    
def get_voxel_size(p_path):
    settings_path = p_path / "scan settings.txt"
    with open(settings_path, "r") as settings_file:
        lines = settings_file.readlines()
    return extract_value(lines, "Voxel size")/1000
 
if __name__ == "__main__":
    base_path = get_data_folder()
    recons_path = base_path / "recons_bh_corr_registered_crop"
    dms_path = base_path / "recons_bh_corr_registered_crop_dm"
    projections_folder = base_path / "projections"
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    com_path = experiment_folder / "centers_of_mass.csv"
    
    with open(com_path, "w") as com_file:
        com_file.write("apple_nr,com_0,com_1,com_2,core_com_0,core_com_1,core_com_2\n")
    
    for apple_nr in range(1, 87):
        apple_path = recons_path / f"{apple_nr}"
        if not apple_path.exists():
            continue
        
        days = sorted([f.name for f in apple_path.iterdir()])
        day = days[0]
        print(day)
        r_path = apple_path / day
        recon = ceu.load_stack(r_path)
        mask = segment_apple(recon, 0.4)
        com = scipy.ndimage.center_of_mass(mask)
        core_mask = segment_core(recon, int(com[0]))
        com_core = scipy.ndimage.center_of_mass(core_mask)
        
        with open(com_path, "a") as com_file:
            com_file.write(f"{apple_nr},{com[0]},{com[1]},{com[2]},{com_core[0]},{com_core[1]},{com_core[2]}\n")
        
        voxel_size = get_voxel_size(projections_folder / day / f"{apple_nr}")
        signed_distance_map = (np.floor(-scipy.ndimage.distance_transform_edt(mask)*voxel_size)
            + np.ceil(scipy.ndimage.distance_transform_edt(~mask)*voxel_size)
            ).astype(np.int8)
        signed_distance_map[core_mask] = -100
        
        ceu.save_stack(dms_path / f"{apple_nr}" / day, signed_distance_map, parents=True)
