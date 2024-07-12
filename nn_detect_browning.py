from pathlib import Path
import os
import sys
import math
from datetime import timedelta
import torch
import tensorboard
import tifffile
import numpy as np
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, resnet101
import pytorch_lightning as pl
import albumentations as A
import albumentations.pytorch as Ap
from matplotlib import pyplot as plt
import cv2

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder

class LRegressionModule(pl.LightningModule):
    def __init__(self, padding_mode, label_noise):
        super().__init__()
        self.model = efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.label_noise = label_noise
        
        convert_conv_to_grayscale_(self.model.features[0][0])
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_ftrs, 1)
        
        # Set padding_mode
        for n, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                if hasattr(m, 'padding_mode'):
                    setattr(m, 'padding_mode', padding_mode)

        self.loss_mse = torch.nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)*3
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        out = self.model(x)*3
        goal = y.to(dtype=out.dtype)
        if self.label_noise:
            goal += torch.rand((1)).to(device=goal.device)-0.5

        mse = self.loss_mse(out[:, 0], goal)
        mae = torch.mean(torch.abs(out[:, 0]-goal))
        maer = torch.mean(torch.abs(torch.round(out[:, 0])-goal))
        
        self.log('mse_train', mse, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('mae_train', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('maer_train', maer, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        return mae
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)*3

        mse = self.loss_mse(out[:, 0], y.to(dtype=out.dtype))
        mae = torch.mean(torch.abs(out[:, 0]-y.to(dtype=out.dtype)))
        maer = torch.mean(torch.abs(torch.round(out[:, 0])-y).to(dtype=torch.float32))
        
        self.log('mse_val', mse, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('mae_val', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('maer_val', maer, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        return mae

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.05)
        return optimizer
    

class LabeledCT():
    def __init__(self, path, label, num_slices, apple_nr, center_slice):
        self.path = path
        self.label = label
        self.num_slices = num_slices
        self.apple_nr = apple_nr
        self.center_slice = center_slice
        
def parse_metadata(metadata_path, recons_path, core_slice_nrs, selected_days, selected_nrs=None, day_dict=None):
    with open(metadata_path) as file:
        metadata = [line.rstrip().split(",") for line in file][1:]
    
    labeled_cts = []
    for record in metadata:
        if record[7] not in selected_days:
            continue
        if day_dict is None:
            day = record[7]
        else:
            day = day_dict[record[7]]
    
        apple_nr = int(record[0][1:])
        if selected_nrs is not None and not apple_nr in selected_nrs:
            continue
        
        path = recons_path / f"{apple_nr}" / day
        if not path.exists():
            print(f"Day not available for apple {apple_nr}")
            continue
        
        label = int(record[5])-1
        num_slices = len(list(path.iterdir()))
        center_slice = core_slice_nrs[apple_nr]
        labeled_cts.append(LabeledCT(path, label, num_slices, apple_nr, center_slice))
    return labeled_cts
    
def parse_core_slice_nrs(centers_of_mass_path):
    data = np.genfromtxt(centers_of_mass_path, delimiter=",", skip_header=1, usecols=[0, 4])
    core_slice_nrs = {}
    for i in range(data.shape[0]):
        core_slice_nrs[int(data[i, 0])] = int(round(data[i, 1]))
    return core_slice_nrs

def load_slice_centered(img_shape, path, path_mask=None, not_mask_value=0, exclude_core=False):
    img = tifffile.imread(path)
    if path_mask is not None:
        dm = tifffile.imread(path_mask)
        mask = dm < 0
        if exclude_core:
            mask = np.logical_xor(mask, dm == -100)
        img[np.logical_not(mask)] = not_mask_value
    result = np.ones(img_shape, dtype=img.dtype)*not_mask_value
    start = (np.array(img_shape)-img.shape)//2
    result[start[0]:start[0]+img.shape[0], start[1]:start[1]+img.shape[1]] = img
    return result

class AugmentedSliceDataset(torch.utils.data.Dataset):

    def __init__(self, recons_path, masks_path, metadata_path, core_slice_nrs, selected_days, day_dict, selected_nrs, region_size, recons_mean, recons_std, apply_augmentations=True, random_slices=True, num_slices=1):
        self.recons_mean = recons_mean
        self.recons_std = recons_std
        self.region_size = region_size
        self.recons_path = recons_path
        self.masks_path = masks_path
        self.labeled_cts = parse_metadata(metadata_path, recons_path, core_slice_nrs, selected_days, selected_nrs, day_dict)
        self.apply_augmentations = apply_augmentations
        self.random_slices = random_slices
        self.num_slices = num_slices

        self.aug_pre = A.Compose([
            A.Flip(p=1),
            A.RandomRotate90(p=1),
            A.Affine(rotate=(-45, 45), scale=(0.95, 1.05), shear=(-10, 10), translate_px=(-10, 10), mode=cv2.BORDER_CONSTANT, cval=0, p=1),
            A.OneOf([
                A.ElasticTransform(alpha=2000, sigma=45, alpha_affine=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.25),
                A.ElasticTransform(alpha=1000, sigma=35, alpha_affine=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.25),
                A.ElasticTransform(alpha=500, sigma=25, alpha_affine=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.25)
            ], p=0.75)
            ])
        self.aug_post = A.Compose([
            A.CropAndPad(px=1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
            A.Normalize(mean=self.recons_mean, std=self.recons_std, max_pixel_value=1.0),
            Ap.ToTensorV2()
            ])
        
    def get_item(self, index):
        scan = self.labeled_cts[index//self.num_slices]
        min_slice_nr = round(scan.center_slice-(self.region_size/2*scan.num_slices))
        max_slice_nr = round(scan.center_slice+(self.region_size/2*scan.num_slices))
        if self.random_slices:
            slice_nr = torch.randint(min_slice_nr, max_slice_nr, (1,)).item()
        else:
            subindex = (index % self.num_slices)
            slice_nr = round((subindex+0.5)*(max_slice_nr-min_slice_nr)/self.num_slices+min_slice_nr)
            
        recon_slice_path = scan.path / f"output{slice_nr:05d}.tif"
        mask_slice_path = next((self.masks_path / str(scan.apple_nr)).iterdir()) / recon_slice_path.name
        img = load_slice_centered((688, 688), recon_slice_path, mask_slice_path, 0)
        if self.apply_augmentations:
            img = self.aug_pre(image=img)["image"]
            # Albumentations' A.GaussNoise clips the input to a maximum value
            # of 1. Therefore noise is added outside of Albumentations.
            if np.random.uniform(0, 1) > 0.5:
                img += np.random.normal(0, np.random.uniform(0, 0.1), img.shape)
        
        aug_data = self.aug_post(image=img)
        
        return aug_data["image"], scan.label
        
    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.labeled_cts)*self.num_slices
    
class FlipRotateSliceDataset(torch.utils.data.Dataset):
    def __init__(self, recons_path, masks_path, metadata_path, core_slice_nrs, selected_days, day_dict, selected_nrs, region_size, recons_mean, recons_std, num_slices=1):
        self.recons_mean = recons_mean
        self.recons_std = recons_std
        self.region_size = region_size
        self.recons_path = recons_path
        self.masks_path = masks_path
        self.labeled_cts = parse_metadata(metadata_path, recons_path, core_slice_nrs, selected_days, selected_nrs, day_dict)
        self.num_slices = num_slices

        self.augmentations = A.Compose([
            A.CropAndPad(px=1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
            A.Normalize(mean=self.recons_mean, std=self.recons_std, max_pixel_value=1.0),
            Ap.ToTensorV2()
            ])
        self.load_slices()
        
    def load_slices(self):
        self.slices = []
        for scan in self.labeled_cts:
            min_slice_nr = round(scan.center_slice-(self.region_size/2*scan.num_slices))
            max_slice_nr = round(scan.center_slice+(self.region_size/2*scan.num_slices))
            for subindex in range(self.num_slices):
                slice_nr = round((subindex+0.5)/self.num_slices*(max_slice_nr-min_slice_nr)+min_slice_nr)
                
                recon_slice_path = scan.path / f"output{slice_nr:05d}.tif"
                mask_slice_path = next((self.masks_path / str(scan.apple_nr)).iterdir()) / recon_slice_path.name
                img = load_slice_centered((688, 688), recon_slice_path, mask_slice_path, 0)
                aug_data = self.augmentations(image=img)
                self.slices.append((aug_data["image"], scan.label))
            
        
    def get_item(self, index):
        img, label = self.slices[index]
        subindex = index % 8
        img2 = torch.rot90(img, k=subindex//2, dims=[1, 2])
        if subindex % 2 == 0:
            img2 = torch.flip(img2, dims=[1])
        
        return img2, label
        
    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.slices)


def convert_conv_to_grayscale_(layer):
    layer.weight.data = layer.weight.data.sum(axis=1, keepdim=True)
    
def train_regression(experiment_folder, l_module, train_loader, val_loader):    
    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "mae_train",
        filename="best_mae_train_{epoch}",
        monitor="mae_train",
        save_top_k=1,
        mode="min"))
    
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "mae_val",
        filename="best_mae_val_{epoch}",
        monitor="mae_val",
        save_top_k=5,
        mode="min"))
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "maer_val",
        filename="best_maer_val_{epoch}",
        monitor="maer_val",
        save_top_k=1,
        mode="min"))
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "mse_val",
        filename="best_mse_val_{epoch}",
        monitor="mse_val",
        save_top_k=1,
        mode="min"))
        
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "hour",
        filename="{epoch}_{step}",
        train_time_interval=timedelta(hours=1),
        save_top_k=-1))
    
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=experiment_folder / "logs")
    trainer = pl.Trainer(
        max_epochs=50000,
        accelerator="gpu",
        devices=2,
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks = callbacks)
    trainer.fit(l_module, train_loader, val_loader)

if __name__ == "__main__":
    base_path = get_data_folder()
    recons_path = base_path / "recons_bh_corr_registered_crop"
    masks_path = base_path / "recons_bh_corr_registered_crop_dm"
    metadata_path = base_path / "Brae_browning_score.csv"
    centers_of_mass_path = base_path / "centers_of_mass.csv"
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
    core_slice_nrs = parse_core_slice_nrs(centers_of_mass_path)
    
    # These are the settings for a prediction network
    # If you want train an detection network instead uncomment the two
    # commented days below and set day_dict to None
    selected_days = [
        "2023-01-23 CA storage 10 weeks out 2 weeks",
        #"2023-01-30 CA storage 10 weeks out 3 weeks",
        "2023-02-13 CA storage 13 weeks out day 15",
        #"2023-03-13 CA storage 18 weeks out day 8",
        "2023-03-20 CA storage 18 weeks out day 15"
    ]
    day_dict = {
        "2023-01-23 CA storage 10 weeks out 2 weeks" : "2023-01-09 CA storage 10 weeks",
        "2023-01-30 CA storage 10 weeks out 3 weeks" : "2023-01-09 CA storage 10 weeks",
        "2023-02-13 CA storage 13 weeks out day 15" : "2023-01-30 CA storage 13 weeks out day 1",
        "2023-03-13 CA storage 18 weeks out day 8" : "2023-03-06 CA storage 18 weeks out day 1",
        "2023-03-20 CA storage 18 weeks out day 15" : "2023-03-06 CA storage 18 weeks out day 1"
    }
    #day_dict = None
    train_set_nrs = [41, 42, 50, 49, 16, 19, 24, 28, 33, 45, 51, 53, 54, 55, 56,
                     57, 58, 59, 60, 61, 63, 6, 29, 64, 66, 68, 72, 74, 77, 80,
                     81, 82, 85, 22, 34, 35, 46, 67, 70, 71, 73, 75, 76, 78, 79]
    val_set_nrs = [44, 40, 20, 26, 52, 47, 62, 83, 84, 48, 69, 65, 86, 21, 39]
    
    # Calculated over all included (close to the core) slices of the training set
    # Excluded the background only
    dataset_mean = np.float32(0.7961113591168265)
    dataset_std = np.float32(0.14691202875131731)
    
    
    train_dataset = AugmentedSliceDataset(
        recons_path,
        masks_path,
        metadata_path,
        core_slice_nrs,
        selected_days,
        day_dict,
        train_set_nrs,
        0.2,
        dataset_mean,
        dataset_std)
    val_dataset = FlipRotateSliceDataset(
        recons_path,
        masks_path,
        metadata_path,
        core_slice_nrs,
        selected_days,
        day_dict,
        val_set_nrs,
        0.2,
        dataset_mean,
        dataset_std,
        num_slices=80
        )
    print(f"Training samples = {len(train_dataset)}")
    print(f"Validation samples = {len(val_dataset)}")
    
    train_loader = utils.data.DataLoader(train_dataset, batch_size=11, num_workers=0, shuffle=True)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=10, num_workers=0)
    
    # padding_mode options: "zeros", "reflect", "replicate", and "circular"
    l_module = LRegressionModule(padding_mode="replicate", label_noise=False)
    train_regression(experiment_folder, l_module, train_loader, val_loader)
