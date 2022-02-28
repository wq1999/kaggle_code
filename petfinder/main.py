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
from tqdm import tqdm

gc.enable()
pd.set_option('display.max_columns', None)

# Image Aug
from utils import get_train_transforms, get_valid_transforms
from utils import divice_norm_bias, MetricMonitor, usr_rmse_score

# Deep Learning
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from utils import get_scheduler, validate_fn, train_fn_amp
from dataset import CuteDataset
from models import PetNet
# Metrics
from sklearn.metrics import mean_squared_error

# Random Seed Initialize
RANDOM_SEED = 555


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
    csv_dir = '/home/wq/kaggle/petfinder-pawpularity-score'
    train_dir = '/home/wq/kaggle/petfinder-pawpularity-score/train'
    test_dir = '/home/wq/kaggle/petfinder-pawpularity-score/test'

    train_file_path = '/home/wq/kaggle/petfinder-pawpularity-score/train.csv'
    sample_sub_file_path = os.path.join(csv_dir, 'sample_submission.csv')

    print(f'Train file: {train_file_path}')
    print(f'Train file: {sample_sub_file_path}')

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(sample_sub_file_path)

    def return_filpath(name, folder=train_dir):
        path = os.path.join(folder, f'{name}.jpg')
        return path

    train_df['image_path'] = train_df['Id'].apply(lambda x: return_filpath(x))
    test_df['image_path'] = test_df['Id'].apply(lambda x: return_filpath(x, folder=test_dir))

    # K Folds
    train_df['norm_score'] = train_df['Pawpularity'] / 100
    num_bins = int(np.floor(1+(3.3)*(np.log2(len(train_df)))))
    train_df['bins'] = pd.cut(train_df['norm_score'], bins=num_bins, labels=False)
    train_df['fold'] = -1
    N_FOLDS = 10

    strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_SEED, shuffle=True)
    for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['bins'])):
        train_df.iloc[train_index, -1] = i

    train_df['fold'] = train_df['fold'].astype('int')

    # CFG
    params = {
        'model': 'swin_large_patch4_window7_224_in22k',
        'pretrained': True,
        'inp_channels': 3,
        'im_size': 224,
        'device': device,
        'lr': 1e-5,
        'use_amp': True,
        'weight_decay': 1e-6,
        'batch_size': 32,
        'num_workers': 8,
        'epochs': 5,
        'out_features': 1,
        'dropout': 0.2,
        'num_fold': N_FOLDS,
        'scheduler_name': 'OneCycleLR',
        'T_0': 5,
        'T_max': 5,
        'T_mult': 1,
        'min_lr': 1e-7,
        'max_lr': 1e-5,
        'opt_wd_non_norm_bias': 0.01,
        'opt_wd_norm_bias': 0,
        'opt_beta1': 0.9,
        'opt_beta2': 0.99,
        'opt_eps': 1e-5,
        'epoch_step_valid': 3,
        'steps_per_epoch': 32
    }

    # Run
    best_models_of_each_fold = []
    rmse_tracker = []

    val_pred = []
    val_target = []

    save_dir = '/home/wq/kaggle_output/petfinder_pytorch'
    os.makedirs(save_dir, exist_ok=True)

    store_path = os.path.join(save_dir, params['model'])
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
        y_train = train['norm_score']

        X_valid = valid['image_path']
        y_valid = valid['norm_score']

        # Pytorch Dataset Creation
        train_dataset = CuteDataset(
            images_filepaths=X_train.values,
            targets=y_train.values,
            transform=get_train_transforms(params=params)
        )

        valid_dataset = CuteDataset(
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
        model = PetNet(params=params)
        model = model.to(params['device'])
        criterion = nn.BCEWithLogitsLoss()
        norm_bias_params, non_norm_bias_params = divice_norm_bias(model)
        optimizer = torch.optim.AdamW(
            [
                {'params': norm_bias_params, 'weight_decay': params['opt_wd_norm_bias']},
                {'params': non_norm_bias_params, 'weight_decay': params['opt_wd_non_norm_bias']},
            ],
            betas=(params['opt_beta1'], params['opt_beta2']),
            eps=params['opt_eps'],
            lr=params['lr'],
            amsgrad=False
        )

        # optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
        #                               weight_decay=params['weight_decay'],
        #                               amsgrad=False)

        scheduler = get_scheduler(optimizer, scheduler_params=params, train_df=train_df)

        # Training and Validation Loop
        best_rmse = np.inf
        best_model_name = None
        for epoch in range(1, params['epochs'] + 1):
            # if params['use_amp']:
            #     scaler = GradScaler()
            #     train_fn_amp(train_loader, model, criterion, optimizer, epoch, params, scaler, scheduler)
            # else:
            #     train_fn(train_loader, model, criterion, optimizer, epoch, params, scheduler)
            # predictions, valid_targets = validate_fn(val_loader, model, criterion, epoch, params)
            # rmse = round(mean_squared_error(valid_targets, predictions, squared=False), 3)

            scaler = GradScaler()
            metric_monitor = MetricMonitor()
            model.train()
            stream = tqdm(train_loader)

            for i, (images, target) in enumerate(stream, start=1):

                images = images.to(params['device'], non_blocking=True)
                target = target.to(params['device'], non_blocking=True).float().view(-1, 1)

                optimizer.zero_grad()

                with autocast():

                    output = model(images)
                    loss = criterion(output, target)

                rmse_score = usr_rmse_score(output, target)
                metric_monitor.update('Loss', loss.item())
                metric_monitor.update('RMSE', rmse_score)

                stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                if (((i % params['steps_per_epoch'] == 0) & (epoch >= params['epoch_step_valid'])) | (
                        i == len(train_loader))):

                    predictions, valid_targets = validate_fn(val_loader, model, criterion, params)
                    rmse = round(mean_squared_error(valid_targets, predictions, squared=False), 3)

                    if rmse < best_rmse:
                        best_rmse = rmse
                        if best_model_name is not None:
                            os.remove(store_path + '/' + best_model_name)
                        torch.save(model.state_dict(),
                                   store_path + f"/{params['model']}_fold{fold + 1}_{best_rmse}.pth")
                        best_model_name = f"{params['model']}_fold{fold + 1}_{best_rmse}.pth"

                    print(f'epoch: {epoch}, batch: {i}/{len(train_loader)}, valid rmse: {rmse}')

        # Print summary of this fold
        print('')
        print(f'The best RMSE: {best_rmse} for fold {fold + 1}.')
        print(f'The Best saved model is: {best_model_name}')

        model.load_state_dict(torch.load(store_path + '/' + best_model_name))
        predictions, valid_targets = validate_fn(val_loader, model, criterion, params)
        rmse = round(mean_squared_error(valid_targets, predictions, squared=False), 3)
        print(rmse)
        val_pred.append(predictions)
        val_target.append(valid_targets)

        best_models_of_each_fold.append(best_model_name)
        rmse_tracker.append(best_rmse)
        print(''.join(['#'] * 50))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print('')
    p = np.concatenate(val_pred)
    t = np.concatenate(val_target)
    rmse = round(mean_squared_error(t, p, squared=False), 3)
    # print(f'Average RMSE of all folds: {round(np.sqrt(np.mean([i ** 2 for i in rmse_tracker])), 4)}')
    print(f'Average RMSE of all folds: {rmse}')

    for i, name in enumerate(best_models_of_each_fold):
        print(f'Best model of fold {i + 1}: {name}')
