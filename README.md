# Longitudinal_ct_workflow
This repository contains the Python code used in the paper (publication in progress) _"Longitudinal CT scanning for explainable early detection of postharvest disorders: The ‘Braeburn’ browning case"_. The paper describes a workflow for acquiring longitudinal CT datasets for developing early detection systems of postharvest disorders. It uses image registration to align the orientations of the CT scans, and four data analysis methods for detecting, analyzing, and understanding the time-dependent evolution of internal disorders.

## Running the code

### Cloning the repository
To clone the repository with submodules use the following command:
```
git clone --recurse-submodules git@github.com:D1rk123/longitudinal_ct_workflow.git
```

### Conda environment
To create the conda environment, follow the instructions in create environment pt22.txt. 

### Folder locations script
To run the scripts you need to create an extra script *folder_locations.py* that contains two functions: get\_data\_folder() and get\_results\_folder(). The path returned by get\_data\_folder() has to contain the data i.e. the CT scans and slice photographs. The results will be saved in the path returned by get\_results\_folder(). For example:
```python
from pathlib import Path

def get_data_folder():
    return Path.home() / "scandata" / "longitudinal_ct_workflow"
    
def get_results_folder():
    return Path.home() / "experiments" / "longitudinal_ct_workflow"
```

### Scripts
Different scripts were used at different parts of the workflow. Here is an overview:

**Longitudinal CT scanning**
- autograd_operator.py
- fit_bh_params_juice.py
- fix_scan_settings.py
- recon_corr_bh_crop.py

**Image registration**
- image_registration_and_crop.py

**Difference Image visualizations**
- plot_difference_image_CT.py
- plot_difference_image_proj.py

**Quantitative analysis of regional changes**
- calc_porosity.py
- plot_porosity_boxplots.py
- plot_segmentation.py
- segment_CT_scans.py

**Classification for early detection**
- apply_NN.py
- apply_NN_proj.py
- calc_constant_performance.py
- calc_train_set_mean_and_std.py
- flatfield_crop.py
- nn_detect_browning.py
- nn_detect_browning_proj.py
- nn_hp_opt_proj.py
- optuna_pl_callback.py
- split_train_test_val.py

**Longitudinally explainable deep learning**
- calc_baselines_performance.py
- invertible_augmentations.py
- plot_longitudinal_IG.py
- plot_longitudinal_IG_proj.py
