import warnings
import sklearn.exceptions
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
import pandas as pd
import numpy as np
import os
import random
import gc

gc.enable()
pd.set_option('display.max_columns', None)

# Image Aug
from utils import get_train_transforms, get_valid_transforms

# Deep Learning
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from utils import get_scheduler, train_fn, validate_fn, train_fn_amp
from dataset import CassavaDataset
from models import CassvaImgClassifier, CassvaImgClassifier_ViT
from losses import FocalLoss
# Metrics
from sklearn.metrics import accuracy_score

# Random Seed Initialize
RANDOM_SEED = 999


def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything()

# Device Optimization
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

if __name__ == '__main__':
    # file path
    csv_dir = '/home/wq/kaggle/cassava-leaf-disease-classification'
    train_dir = '/home/wq/kaggle/cassava-leaf-disease-classification/train_images'
    test_dir = '/home/wq/kaggle/cassava-leaf-disease-classification/test_images'

    train_file_path = '/home/wq/kaggle/cassava-leaf-disease-classification/train.csv'
    sample_sub_file_path = os.path.join(csv_dir, 'sample_submission.csv')

    print(f'Train file: {train_file_path}')
    print(f'Train file: {sample_sub_file_path}')

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(sample_sub_file_path)

    def return_filpath(name, folder=train_dir):
        path = os.path.join(folder, name)
        return path

    train_df['image_path'] = train_df['image_id'].apply(lambda x: return_filpath(x))
    test_df['image_path'] = test_df['image_id'].apply(lambda x: return_filpath(x, folder=test_dir))

    # K Folds
    train_df['fold'] = -1
    N_FOLDS = 5

    strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_SEED, shuffle=True)
    for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['label'])):
        train_df.iloc[train_index, -1] = i

    train_df['fold'] = train_df['fold'].astype('int')

    target = ['label']

    # CFG
    params = {
        'model': 'swin_base_patch4_window12_384_in22k',
        'im_size': 384,
        'device': device,
        'lr': 1e-4,
        'use_amp': True,
        'weight_decay': 1e-6,
        'batch_size': 16,
        'num_workers': 8,
        'epochs': 10,
        'num_fold': N_FOLDS,
        'scheduler_name': 'CosineAnnealingWarmRestarts',
        'T_0': 10,
        'T_max': 5,
        'T_mult': 1,
        'min_lr': 1e-6,
        'max_lr': 1e-4,
    }

    # Run
    best_models_of_each_fold = []
    acc_tracker = []

    save_dir = '/home/wq/kaggle_output/cassave_leaf'
    os.makedirs(save_dir, exist_ok=True)

    store_path = os.path.join(save_dir, params['model'] + '_v2')
    os.makedirs(store_path, exist_ok=True)
    if os.path.exists(store_path):
        pass

    for fold in range(N_FOLDS):
        print(''.join(['#'] * 50))
        print(f"{''.join(['='] * 15)} TRAINING FOLD: {fold + 1}/{train_df['fold'].nunique()} {''.join(['='] * 15)}")
        # Data Split to train and Validation
        train = train_df[train_df['fold'] != fold]
        valid = train_df[train_df['fold'] == fold]

        X_train = train['image_path']
        y_train = train['label']
        X_valid = valid['image_path']
        y_valid = valid['label']

        # Pytorch Dataset Creation
        train_dataset = CassavaDataset(
            images_filepaths=X_train.values,
            targets=y_train.values,
            transform=get_train_transforms(params=params)
        )

        valid_dataset = CassavaDataset(
            images_filepaths=X_valid.values,
            targets=y_valid.values,
            transform=get_valid_transforms(params=params)
        )

        # Pytorch Dataloader creation
        train_loader = DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers'], pin_memory=True
        )

        val_loader = DataLoader(
            valid_dataset, batch_size=params['batch_size'], shuffle=False,
            num_workers=params['num_workers'], pin_memory=True
        )

        # Model, cost function and optimizer instancing
        model = CassvaImgClassifier_ViT(params['model'], train.label.nunique(), pretrained=True)
        model = model.to(params['device'])
        criterion = FocalLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                      weight_decay=params['weight_decay'],
                                      amsgrad=False)
        scheduler = get_scheduler(optimizer, scheduler_params=params, train_df=train_df)

        # Training and Validation Loop
        best_acc = 0.
        best_epoch = np.inf
        best_model_name = None
        for epoch in range(1, params['epochs'] + 1):
            if params['use_amp']:
                scaler = GradScaler()
                train_fn_amp(train_loader, model, criterion, optimizer, epoch, params, scaler, scheduler)
            else:
                train_fn(train_loader, model, criterion, optimizer, epoch, params, scheduler)
            predictions, valid_targets = validate_fn(val_loader, model, criterion, epoch, params)
            acc = round(accuracy_score(valid_targets, predictions), 3)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                if best_model_name is not None:
                    os.remove(store_path + '/' + best_model_name)
                torch.save(model.state_dict(),
                           store_path + f"/{params['model']}_{epoch}_epoch_f{fold + 1}_{acc}_acc.pth")
                best_model_name = f"{params['model']}_{epoch}_epoch_f{fold + 1}_{acc}_acc.pth"

        # Print summary of this fold
        print('')
        print(f'The best Acc: {best_acc} for fold {fold + 1} was achieved on epoch: {best_epoch}.')
        print(f'The Best saved model is: {best_model_name}')
        best_models_of_each_fold.append(best_model_name)
        acc_tracker.append(best_acc)
        print(''.join(['#'] * 50))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print('')
    print(f'Average Acc of all folds: {round(np.mean(acc_tracker), 4)}')

    for i, name in enumerate(best_models_of_each_fold):
        print(f'Best model of fold {i + 1}: {name}')
