import warnings
import sklearn.exceptions
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
import pandas as pd
import numpy as np
import os, glob
import time
import random
import gc
from tqdm import tqdm

gc.enable()
pd.set_option('display.max_columns', None)

# Image Aug
from utils import get_train_transforms, get_valid_transforms

# Deep Learning
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from utils import train_fn, validate_fn, train_fn_amp
from dataset import CassavaDataset, CassavaTestDataset
from models import CassvaImgClassifier, CassvaImgClassifier_ViT
from losses import FocalLoss
# Metrics
from sklearn.metrics import f1_score

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

# Device Optimization
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

if __name__ == '__main__':
    # file path
    train_dir = '/home/wq/kaggle/car_data/train/train'
    train_file_path = '/home/wq/kaggle/car_data/train/train_sorted.csv'

    label2idx = {'car': 0, 'suv': 1, 'van': 2, 'truck': 3}
    idx2label = {v: k for k, v in label2idx.items()}

    print(label2idx)
    print(idx2label)

    print(f'Train file: {train_file_path}')

    train_df = pd.read_csv(train_file_path)

    def return_filpath(name, folder=train_dir):
        path = os.path.join(folder, name)
        return path

    train_df['image_path'] = train_df['id'].apply(lambda x: return_filpath(x))
    train_df['type'] = train_df['type'].map(label2idx)

    print(train_df)

    # K Folds
    train_df['fold'] = -1
    N_FOLDS = 10

    strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_SEED, shuffle=True)
    for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['type'])):
        train_df.iloc[train_index, -1] = i

    train_df['fold'] = train_df['fold'].astype('int')

    # CFG
    params = {
        'model': 'swin_base_patch4_window7_224', # swin_base_patch4_window7_224
        'im_size': 224,
        'device': device,
        'lr': 1e-4,
        'use_amp': True,
        'weight_decay': 1e-6,
        'batch_size': 32,
        'num_workers': 8,
        'epochs': 50,
        'num_fold': N_FOLDS,
    }

    # Run
    best_models_of_each_fold = []
    acc_tracker = []

    val_pred = []
    val_target = []

    save_dir = './'
    os.makedirs(save_dir, exist_ok=True)

    store_path = os.path.join(save_dir, params['model'] + '_V1')
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
        y_train = train['type']
        X_valid = valid['image_path']
        y_valid = valid['type']

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
        model = CassvaImgClassifier_ViT(params['model'], train.type.nunique(), pretrained=True)
        model = model.to(params['device'])
        criterion = FocalLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                      weight_decay=params['weight_decay'],
                                      amsgrad=False)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params['lr'],
                                                        steps_per_epoch=len(train_loader),
                                                        epochs=params['epochs'])

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
            acc = round(f1_score(valid_targets, predictions, average='macro'), 3)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                if best_model_name is not None:
                    os.remove(store_path + '/' + best_model_name)
                torch.save(model.state_dict(),
                           store_path + f"/{params['model']}_{epoch}_epoch_f{fold + 1}_{acc}_f1_score.pth")
                best_model_name = f"{params['model']}_{epoch}_epoch_f{fold + 1}_{acc}_f1_score.pth"

        # Print summary of this fold
        print('')
        print(f'The best F1 Score: {best_acc} for fold {fold + 1} was achieved on epoch: {best_epoch}.')
        print(f'The Best saved model is: {best_model_name}')

        model.load_state_dict(torch.load(store_path + '/' + best_model_name))
        predictions, valid_targets = validate_fn(val_loader, model, criterion, best_epoch, params)
        f1 = round(f1_score(valid_targets, predictions, average='macro'), 3)
        print(f1)
        val_pred.append(predictions)
        val_target.append(valid_targets)

        best_models_of_each_fold.append(best_model_name)
        acc_tracker.append(best_acc)
        print(''.join(['#'] * 50))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print('')
    p = np.concatenate(val_pred)
    t = np.concatenate(val_target)
    f1 = round(f1_score(t, p, average='macro'), 3)
    print(f'Average F1 Score of all folds: {f1}')

    for i, name in enumerate(best_models_of_each_fold):
        print(f'Best model of fold {i + 1}: {name}')

    # test
    root = '/home/wq/kaggle/car_data/testA/'
    images = glob.glob(os.path.join(root, '*.jpg'))
    name2idx = {}
    for i in range(len(images)):
        idx = int(images[i].split('/')[-1].split('.')[0].split('_')[-1])
        name2idx[idx] = images[i].split('/')[-1]
    name2idx = sorted(name2idx.items(), key=lambda item: item[0])
    images = []
    for i in name2idx:
        images.append(root + i[1])
    data_test = pd.DataFrame(index=range(len(images)))
    data_test['file'] = images
    print(data_test)

    test_dataset = CassavaTestDataset(data_test['file'], transform=get_valid_transforms(params=params))
    test_data_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

    # model
    weight_files = glob.glob(store_path + "/*.pth")
    model_lists = []
    for weight_file in weight_files:
        print(f"filename:{weight_file}")

        model = CassvaImgClassifier_ViT(params['model'], 4, pretrained=False)
        model = model.to(params['device'])
        net_weights = torch.load(weight_file)
        model.load_state_dict(net_weights)
        model.eval()

        model_lists.append(model)

    # infer
    ttar = []
    with torch.no_grad():
        result = []
        for image, filename in tqdm(test_data_loader):
            output = 0.
            for model in model_lists:
                output += model(image.cuda())

            output = output / len(model_lists)
            output = nn.functional.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)
            output = output.cpu().numpy()

            for i in range(len(output)):
                pred = idx2label[output[i]]
                result.append([filename[i].split('.')[-2].split('/')[-1] + '.jpg', pred])

        result = pd.DataFrame(np.array(result), columns=["id", "type"])
        ttar.append(result)

    r = ttar[0]

    tm = str(time.strftime('%m%d%H%M'))
    wc = params['model'] + '_10folds-val_acc={}'.format(f1)
    r.to_csv(store_path + "/submit_{}_{}.csv".format(wc, tm), index=False)
