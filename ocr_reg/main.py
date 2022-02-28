import warnings
import sklearn.exceptions
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
import pandas as pd
import numpy as np
import os, glob
import random
import gc
import argparse

gc.enable()
pd.set_option('display.max_columns', None)

# Image Aug
from utils import get_train_transforms, get_valid_transforms

# Deep Learning
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from utils import validate_fn, rand_bbox, MetricMonitor, calculat_acc
from dataset import CaptchaData
from models import Net
from tqdm import tqdm
from torch.cuda.amp import autocast

# Random Seed Initialize
RANDOM_SEED = 42


def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything()


def parse_opts():

    parser = argparse.ArgumentParser()

    # args
    parser.add_argument('--model', type=str, default='tf_efficientnet_b5_ns')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step', type=int, default=64)
    parser.add_argument('--valid_epoch', type=int, default=80)
    parser.add_argument('--size', type=int, default=160)
    parser.add_argument('--flag', type=int, default=1)
    parser.add_argument('--filename', type=str, default='')
    args = parser.parse_args()
    return args


# python main.py --model tf_efficientnet_b5_ns --epoch 150 --step 64 --valid_epoch 120 --size 200 --flag 0 --filename raw_data


# Device Optimization
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')


if __name__ == '__main__':
    # parameters
    args = parse_opts()
    model = args.model
    epoch = args.epoch
    step = args.step
    valid_epoch = args.valid_epoch
    size = args.size
    flag = args.flag
    filename = args.filename

    print('Use model {}'.format(model))

    if flag == 1:
        use_ext = True
    else:
        use_ext = False

    # config
    params = {
        'model': model,
        'im_size': size,
        'device': device,
        'lr': 1e-3,
        'use_amp': True,
        'use_ext': use_ext,
        'weight_decay': 1e-6,
        'batch_size': 32,
        'num_workers': 8,
        'epochs': epoch,
        'num_fold': 5,
        'steps_per_epoch': step,
        'epoch_step_valid': valid_epoch,
    }

    # file path
    root = '/home/wq/kaggle/ocr_data/training_dataset'
    images = glob.glob(os.path.join(root, '*.png'))
    labels = [image.split('.')[-2].split('/')[-1] for image in images]
    train = pd.DataFrame(index=range(len(images)))
    train['filename'] = images
    train['label'] = labels
    # ext data
    root_ext = '/home/wq/kaggle/ext-data/captcha_ext'
    images = glob.glob(os.path.join(root_ext, '*.png'))
    labels = [image.split('.')[-2].split('/')[-1] for image in images]
    train_ext = pd.DataFrame(index=range(len(images)))
    train_ext['filename'] = images
    train_ext['label'] = labels
    # concat
    if params['use_ext']:
        print('Use External Data')
        train = pd.concat([train, train_ext])
    else:
        print('Use Raw Data')

    # K Folds
    train['fold'] = -1
    N_FOLDS = params['num_fold']

    strat_kfold = KFold(n_splits=N_FOLDS, random_state=RANDOM_SEED, shuffle=True)
    for i, (_, train_index) in enumerate(strat_kfold.split(train.index)):
        train.iloc[train_index, -1] = i

    train['fold'] = train['fold'].astype('int')

    # Run
    source = [str(i) for i in range(0, 10)]
    source += [chr(i) for i in range(97, 97 + 26)]
    source += [chr(i) for i in range(65, 65 + 26)]
    alphabet = ''.join(source)

    best_models_of_each_fold = []
    acc_tracker = []

    save_dir = './'
    os.makedirs(save_dir, exist_ok=True)

    store_path = os.path.join(save_dir, params['model'] + '-' + filename)
    os.makedirs(store_path, exist_ok=True)

    for fold in range(N_FOLDS):

        if params['model'] == 'tf_efficientnet_b7_ns':
            if fold != 0:
                continue

        print(''.join(['#'] * 50))
        print(f"{''.join(['='] * 15)} TRAINING FOLD: {fold + 1}/{train['fold'].nunique()} {''.join(['='] * 15)}")
        # Data Split to train and Validation
        X_train = train[train['fold'] != fold]
        X_valid = train[train['fold'] == fold]

        train_dataset = CaptchaData(X_train, transform=get_train_transforms(params=params), alphabet=alphabet)
        valid_dataset = CaptchaData(X_valid, transform=get_valid_transforms(params=params), alphabet=alphabet)

        train_data_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=params['batch_size'] * 2)

        # Model, cost function and optimizer instancing
        model = Net(params['model'], 62 * 4, pretrained=True)
        model = model.to(params['device'])

        criterion = nn.MultiLabelSoftMarginLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                      weight_decay=params['weight_decay'],
                                      amsgrad=False)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params['lr'],
                                                        steps_per_epoch=len(train_data_loader),
                                                        epochs=params['epochs'])

        # Training and Validation Loop
        best_acc = 0.
        best_epoch = None
        best_model_name = None
        for epoch in range(1, params['epochs'] + 1):

            scaler = GradScaler()
            metric_monitor = MetricMonitor()
            model.train()
            stream = tqdm(train_data_loader)

            beta = 1
            alpha = 0.4
            cutmix_prob = 0.5

            for i, (images, target) in enumerate(stream, start=1):

                img = images.to(params['device'], non_blocking=True).float()
                target = target.to(params['device'], non_blocking=True).long()

                optimizer.zero_grad()

                with autocast():

                    r = np.random.rand(1)
                    if beta > 0 and r < cutmix_prob:
                        # cutmix
                        # generate mixed sample
                        lam = np.random.beta(beta, beta)
                        rand_index = torch.randperm(img.size()[0]).cuda()
                        target_a = target
                        target_b = target[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
                        img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
                        # adjust lambda to exactly match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
                        # compute output
                        output = model(img)
                        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                    else:
                        # mixup
                        lam = np.random.beta(alpha, alpha)
                        index = torch.randperm(img.size()[0]).cuda()
                        mixed_x = lam * img + (1 - lam) * img[index, :]
                        y_a, y_b = target, target[index]
                        output = model(mixed_x)
                        loss = lam * criterion(output, y_a) + (1. - lam) * criterion(output, y_b)

                acc = calculat_acc(output, target)
                metric_monitor.update('Loss', loss.item())
                metric_monitor.update('Acc', acc)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")

                if scheduler is not None:
                    scheduler.step()

                if (((i % params['steps_per_epoch'] == 0) & (epoch > params['epoch_step_valid'])) | (
                        i == len(train_data_loader))):

                    acc = validate_fn(valid_data_loader, model, criterion, epoch, params)

                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = epoch
                        if best_model_name is not None:
                            os.remove(store_path + '/' + best_model_name)
                        torch.save(model.state_dict(),
                                   store_path + '/' + params['model'] + "_fold{}-val_acc={}.pth".format(fold + 1,
                                                                                                        best_acc))
                        best_model_name = params['model'] + "_fold{}-val_acc={}.pth".format(fold + 1, best_acc)

                    print(f'epoch: {epoch}, batch: {i}/{len(train_data_loader)}, valid acc: {acc}')

                    model.train()

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
