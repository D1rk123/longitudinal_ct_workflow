import torch
import tifffile
import numpy as np
from torch import optim, utils
from torchvision.models import efficientnet_v2_s
import pytorch_lightning as pl
import albumentations as A
import albumentations.pytorch as Ap
import cv2
import optuna
from optuna_pl_callback import PyTorchLightningPruningCallback
import functools
from getpass import getpass
import sys

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder

class LRegressionModule(pl.LightningModule):
    def __init__(self, padding_mode, label_noise, trial):
        super().__init__()
        self.model = efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.label_noise = label_noise
        self.trial = trial
        
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
        weight_decay = self.trial.suggest_float("weight_decay", 0.05, 0.5, log=True)
        optimizer = optim.AdamW(self.parameters(), lr=2e-5, weight_decay=weight_decay)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.5)
        #optimizer = optim.RMSprop(self.parameters(), lr=2e-6, alpha=0.9, weight_decay=1e-4, momentum=0.9)
        return optimizer
    

class LabeledPStack():
    def __init__(self, path, label, num_projs, apple_nr):
        self.path = path
        self.label = label
        self.num_projs = num_projs
        self.apple_nr = apple_nr
        
def parse_metadata_proj(metadata_path, ff_proj_path, selected_days, selected_nrs=None, day_dict=None):
    with open(metadata_path) as file:
        metadata = [line.rstrip().split(",") for line in file][1:]
    
    labeled_pstacks = []
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
        
        path = ff_proj_path / day / f"{apple_nr}_ff"
        if not path.exists():
            print(f"Day not available for apple {apple_nr}")
            continue
        
        label = int(record[5])-1
        num_projs = len(list(path.iterdir()))
        labeled_pstacks.append(LabeledPStack(path, label, num_projs, apple_nr))
    return labeled_pstacks

class AugmentedProjDataset(torch.utils.data.Dataset):

    def __init__(self, ff_proj_path, metadata_path, selected_days, day_dict, selected_nrs, ff_proj_mean, ff_proj_std, crop_top, crop_bottom, trial, apply_augmentations=True):
        self.ff_proj_mean = ff_proj_mean
        self.ff_proj_std = ff_proj_std
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.ff_proj_path = ff_proj_path
        self.labeled_pstacks = parse_metadata_proj(metadata_path, ff_proj_path, selected_days, selected_nrs, day_dict)
        self.apply_augmentations = apply_augmentations
        self.trial = trial
        
        self.max_noise_std = self.trial.suggest_float("max_noise_std", 0.002, 0.1, log=True)
        self.max_rotate = self.trial.suggest_float("max_rotate", 10, 45)
        self.max_scale = self.trial.suggest_float("max_scale", 1, 1.2)
        self.max_shear = self.trial.suggest_float("max_shear", 0, 25)
        self.max_translate = self.trial.suggest_int("max_translate", 0, 20)
        self.elastic_alpha_base = self.trial.suggest_float("elastic_alpha_base", 100, 700)
        self.elastic_sigma = self.trial.suggest_float("elastic_sigma", 5, 25)

        self.aug_pre = A.Compose([
            A.HorizontalFlip(p=1),
            A.Affine(
                rotate=(-self.max_rotate, self.max_rotate),
                scale=(1/self.max_scale, self.max_scale),
                shear=(-self.max_shear, self.max_shear),
                translate_px=(-self.max_translate, self.max_translate),
                mode=cv2.BORDER_CONSTANT, cval=0, p=1),
            A.OneOf([
                A.ElasticTransform(
                    alpha=self.elastic_alpha_base,
                    sigma=self.elastic_sigma+self.elastic_alpha_base*0.03,
                    alpha_affine=0,
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=0.25),
                A.ElasticTransform(
                    alpha=self.elastic_alpha_base*2,
                    sigma=self.elastic_sigma+self.elastic_alpha_base*0.03,
                    alpha_affine=0,
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=0.25),
            ], p=0.6667),
            A.Crop(x_min=0, x_max=690, y_min=self.crop_top, y_max=690-self.crop_bottom)
            ])
        self.aug_post = A.Compose([
            A.CropAndPad(px=-1, keep_size=False),
            A.CropAndPad(px=1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
            A.Normalize(mean=self.ff_proj_mean, std=self.ff_proj_std, max_pixel_value=1.0),
            Ap.ToTensorV2()
        ])
        
    def get_item(self, index):
        pstack = self.labeled_pstacks[index]
        proj_nr = torch.randint(0, pstack.num_projs, (1,)).item()
            
        proj_path = pstack.path / f"output{proj_nr:05d}.tif"
        img = tifffile.imread(proj_path)
        if self.apply_augmentations:
            img = self.aug_pre(image=img)["image"]
            # Albumentations' A.GaussNoise clips the input to a maximum value
            # of 1. Therefore noise is added outside of Albumentations.
            if np.random.uniform(0, 1) > 0.5:
                img += np.random.normal(0, np.random.uniform(0, self.max_noise_std), img.shape)
        
        img = self.aug_post(image=img)["image"]
        torch.clamp_(img, -3, 3)
        
        return img, pstack.label
        
    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.labeled_pstacks)
    
class SampleProjDataset(torch.utils.data.Dataset):
    def __init__(self, ff_proj_path, metadata_path, selected_days, day_dict, selected_nrs, ff_proj_mean, ff_proj_std, crop_top, crop_bottom, num_projs=1):
        self.ff_proj_mean = ff_proj_mean
        self.ff_proj_std = ff_proj_std
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.ff_proj_path = ff_proj_path
        self.num_projs = num_projs
        self.labeled_pstacks = parse_metadata_proj(metadata_path, ff_proj_path, selected_days, selected_nrs, day_dict)

        self.augmentations = A.Compose([
            A.Crop(x_min=0, x_max=690, y_min=self.crop_top, y_max=690-self.crop_bottom),
            A.CropAndPad(px=-1, keep_size=False),
            A.CropAndPad(px=1, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False),
            A.Normalize(mean=self.ff_proj_mean, std=self.ff_proj_std, max_pixel_value=1.0),
            Ap.ToTensorV2()
            ])
        self.load_projs()
        
    def load_projs(self):
        self.projs = []
        for pstack in self.labeled_pstacks:
            for proj_nr in np.round(np.linspace(0, pstack.num_projs, self.num_projs, endpoint=False)).astype(int):
                img = tifffile.imread(pstack.path  / f"output{proj_nr:05d}.tif")
                aug_data = self.augmentations(image=img)
                self.projs.append((aug_data["image"], pstack.label))
            
        
    def get_item(self, index):
        return self.projs[index]
        
    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.projs)


def convert_conv_to_grayscale_(layer):
    layer.weight.data = layer.weight.data.sum(axis=1, keepdim=True)
    
def train_regression(experiment_folder, l_module, train_loader, val_loader, trial):    
    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "mae_train",
        filename="best_mae_train_{epoch}_{mae_train:.10f}",
        monitor="mae_train",
        save_top_k=1,
        mode="min"))
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "mae_val",
        filename="best_mae_val_{epoch}_{mae_val:.10f}",
        monitor="mae_val",
        save_top_k=1,
        mode="min"))
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=experiment_folder / "checkpoints" / "maer_val",
        filename="best_maer_val_{epoch}_{maer_val:.10f}",
        monitor="maer_val",
        save_top_k=1,
        mode="min"))
    callbacks.append(pl.callbacks.EarlyStopping(
        monitor = "mae_val",
        patience = 200
        ))
    optuna_callback = PyTorchLightningPruningCallback(trial, monitor="mae_val")
    callbacks.append(optuna_callback)
    
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=experiment_folder / "logs")
    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="gpu",
        devices=2,
        strategy="ddp_spawn",
        logger=tb_logger,
        log_every_n_steps=1,
        callbacks = callbacks)
    trainer.fit(l_module, train_loader, val_loader)
    optuna_callback.check_pruned()

def objective(trial, experiment_folder):
    trial_folder = experiment_folder / f"trial_{trial.number}"
    trial_folder.mkdir()
    
    base_path = get_data_folder()
    ff_proj_path = base_path / "projections_flat_crop"
    metadata_path = base_path / "Brae_browning_score.csv"
    
    selected_days = [
        "2023-01-23 CA storage 10 weeks out 2 weeks",
        "2023-01-30 CA storage 10 weeks out 3 weeks",
        "2023-02-13 CA storage 13 weeks out day 15",
        "2023-03-13 CA storage 18 weeks out day 8",
        "2023-03-20 CA storage 18 weeks out day 15"
    ]
    day_dict = {
        "2023-01-23 CA storage 10 weeks out 2 weeks" : "2023-01-09 CA storage 10 weeks",
        "2023-01-30 CA storage 10 weeks out 3 weeks" : "2023-01-09 CA storage 10 weeks",
        "2023-02-13 CA storage 13 weeks out day 15" : "2023-01-30 CA storage 13 weeks out day 1",
        "2023-03-13 CA storage 18 weeks out day 8" : "2023-03-06 CA storage 18 weeks out day 1",
        "2023-03-20 CA storage 18 weeks out day 15" : "2023-03-06 CA storage 18 weeks out day 1"
    }
    day_dict = None
    train_set_nrs = [41, 42, 50, 49, 16, 19, 24, 28, 33, 45, 51, 53, 54, 55, 56,
                     57, 58, 59, 60, 61, 63, 6, 29, 64, 66, 68, 72, 74, 77, 80,
                     81, 82, 85, 22, 34, 35, 46, 67, 70, 71, 73, 75, 76, 78, 79]
    val_set_nrs = [44, 40, 20, 26, 52, 47, 62, 83, 84, 48, 69, 65, 86, 21, 39]
    
    # Calculated over all projections over all days of the apples of the training set
    dataset_mean = 0.54003301858902
    dataset_std = 0.573314089958484
    
    
    crop_top = trial.suggest_int("crop_top", 0, 200)
    crop_bottom = trial.suggest_int("crop_bottom", 0, 200)
    
    train_dataset = AugmentedProjDataset(
        ff_proj_path,
        metadata_path,
        selected_days,
        day_dict,
        train_set_nrs,
        dataset_mean,
        dataset_std,
        crop_top,
        crop_bottom,
        trial
        )
    val_dataset = SampleProjDataset(
        ff_proj_path,
        metadata_path,
        selected_days,
        day_dict,
        val_set_nrs,
        dataset_mean,
        dataset_std,
        crop_top,
        crop_bottom,
        num_projs=80
        )
    print(f"Training samples = {len(train_dataset)}")
    print(f"Validation samples = {len(val_dataset)}")
    
    train_loader = utils.data.DataLoader(train_dataset, batch_size=11, num_workers=0, shuffle=True)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=10, num_workers=0)
    
    l_module = LRegressionModule(padding_mode="replicate", label_noise=False, trial=trial)
    train_regression(trial_folder, l_module, train_loader, val_loader, trial)
    
    checkpoint_filename = next((trial_folder / "checkpoints" / "mae_val").glob("best_mae_val_*")).name
    best_model_score = float(checkpoint_filename[checkpoint_filename.find("_mae_val=")+9:-5])
    
    return best_model_score

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())

    password = getpass()
    ip = sys.argv[1]
    
    storage = f"mysql://root:{password}@{ip}/optuna"
    study = optuna.create_study(
        study_name="braeburn_browning_proj_detection2",
        storage=storage,
        direction="minimize",
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    
    study.optimize(
        functools.partial(objective, experiment_folder=experiment_folder),
        n_trials=100,
        timeout=None,
        n_jobs=1
    )
    
