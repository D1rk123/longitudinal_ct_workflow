# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:55:50 2023

@author: des
"""
from pathlib import Path
import nn_detect_browning as nndb
import itertools
import random
import numpy as np

def randomly_select_and_remove(amount, day, label, labeled_cts, prefer_first_day):
    pool = [a for a in labeled_cts
        if a.label == label
        and a.path.name == day]
    
    if prefer_first_day:
        fd_pool = [a for a in pool
            if (a.path.parent / "2022-11-02 day 1").exists()]
        non_fd_pool = [a for a in pool
            if not (a.path.parent / "2022-11-02 day 1").exists()]
            
        fd_amount = min(amount, len(fd_pool))
        non_fd_amount = amount - fd_amount 
        selection = random.sample(fd_pool, fd_amount) \
            + random.sample(non_fd_pool, non_fd_amount)
    else:
        selection = random.sample(pool, amount)
    
    for a in selection:
        labeled_cts.remove(a)
        
    return selection

if __name__ == "__main__":
    base_path = Path("/export/scratch2/des/scans_breaburn")
    recons_path = base_path / "recons_bh_corr_registered_crop"
    metadata_path = base_path / "Brae_browning_score.csv"
    
    centers_of_mass_path = base_path / "centers_of_mass.csv"    
    core_slice_nrs = nndb.parse_core_slice_nrs(centers_of_mass_path)
    
    selected_days = [
        "2023-01-23 CA storage 10 weeks out 2 weeks",
        "2023-01-30 CA storage 10 weeks out 3 weeks",
        "2023-02-13 CA storage 13 weeks out day 15",
        "2023-03-13 CA storage 18 weeks out day 8",
        "2023-03-20 CA storage 18 weeks out day 15"
    ]
    
    labeled_cts = nndb.parse_metadata(metadata_path, recons_path, core_slice_nrs, selected_days)
    print(len(labeled_cts))
    
    # Do stratified sampling over both the day and the label
    test_set = (
        randomly_select_and_remove(1, "2023-01-23 CA storage 10 weeks out 2 weeks", 0, labeled_cts, True)
        + randomly_select_and_remove(1, "2023-01-30 CA storage 10 weeks out 3 weeks", 1, labeled_cts, True)
        + randomly_select_and_remove(2, "2023-02-13 CA storage 13 weeks out day 15", 0, labeled_cts, True)
        + randomly_select_and_remove(2, "2023-02-13 CA storage 13 weeks out day 15", 1, labeled_cts, True)
        + randomly_select_and_remove(1, "2023-02-13 CA storage 13 weeks out day 15", 2, labeled_cts, True)
        + randomly_select_and_remove(1, "2023-03-13 CA storage 18 weeks out day 8", 0, labeled_cts, True)
        + randomly_select_and_remove(1, "2023-03-13 CA storage 18 weeks out day 8", 1, labeled_cts, True)
        + randomly_select_and_remove(1, "2023-03-13 CA storage 18 weeks out day 8", 2, labeled_cts, True)
        + randomly_select_and_remove(1, "2023-03-13 CA storage 18 weeks out day 8", 3, labeled_cts, True)
        + randomly_select_and_remove(1, "2023-03-20 CA storage 18 weeks out day 15", 0, labeled_cts, True)
        + randomly_select_and_remove(0, "2023-03-20 CA storage 18 weeks out day 15", 1, labeled_cts, True)
        + randomly_select_and_remove(1, "2023-03-20 CA storage 18 weeks out day 15", 2, labeled_cts, True)
        + randomly_select_and_remove(2, "2023-03-20 CA storage 18 weeks out day 15", 3, labeled_cts, True)
        )
    validation_set = (
        randomly_select_and_remove(1, "2023-01-23 CA storage 10 weeks out 2 weeks", 0, labeled_cts, False)
        + randomly_select_and_remove(1, "2023-01-30 CA storage 10 weeks out 3 weeks", 1, labeled_cts, False)
        + randomly_select_and_remove(2, "2023-02-13 CA storage 13 weeks out day 15", 0, labeled_cts, False)
        + randomly_select_and_remove(2, "2023-02-13 CA storage 13 weeks out day 15", 1, labeled_cts, False)
        + randomly_select_and_remove(1, "2023-02-13 CA storage 13 weeks out day 15", 2, labeled_cts, False)
        + randomly_select_and_remove(1, "2023-03-13 CA storage 18 weeks out day 8", 0, labeled_cts, False)
        + randomly_select_and_remove(1, "2023-03-13 CA storage 18 weeks out day 8", 1, labeled_cts, False)
        + randomly_select_and_remove(1, "2023-03-13 CA storage 18 weeks out day 8", 2, labeled_cts, False)
        + randomly_select_and_remove(1, "2023-03-13 CA storage 18 weeks out day 8", 3, labeled_cts, False)
        + randomly_select_and_remove(1, "2023-03-20 CA storage 18 weeks out day 15", 0, labeled_cts, False)
        + randomly_select_and_remove(0, "2023-03-20 CA storage 18 weeks out day 15", 1, labeled_cts, False)
        + randomly_select_and_remove(1, "2023-03-20 CA storage 18 weeks out day 15", 2, labeled_cts, False)
        + randomly_select_and_remove(2, "2023-03-20 CA storage 18 weeks out day 15", 3, labeled_cts, False)
        )
    # The training set is all remaining samples
        
    test_set_nrs = [a.apple_nr for a in test_set]
    validation_set_nrs = [a.apple_nr for a in validation_set]
    train_set_nrs = [a.apple_nr for a in labeled_cts]
    print(f"{len(test_set_nrs)}: {test_set_nrs}")
    print(f"{len(validation_set_nrs)}: {validation_set_nrs}")
    print(f"{len(train_set_nrs)}: {train_set_nrs}")
    
    test_set_labels = [a.label for a in test_set]
    validation_set_labels = [a.label for a in validation_set]
    train_set_labels = [a.label for a in labeled_cts]
    
    for s in [test_set_labels, validation_set_labels, train_set_labels]:
        labels, counts = np.unique(s, return_counts=True)
        print(counts / np.sum(counts))
