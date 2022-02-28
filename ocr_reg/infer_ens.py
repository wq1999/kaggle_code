import warnings
import sklearn.exceptions

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
import pandas as pd
import numpy as np
import os, glob
import random
import gc

gc.enable()
pd.set_option('display.max_columns', None)

# Image Aug
from utils import get_valid_transforms

# Deep Learning
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from dataset import TestData
from models import Net
from tqdm import tqdm
from torch.autograd import Variable

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
    # CFG
    N_FOLDS = 5
    params = {
        'model1': 'tf_efficientnet_b5_ns',
        'model2': 'tf_efficientnet_b6_ns',
        'im_size': 256,
        'device': device,
        'lr': 1e-3,
        'use_amp': True,
        'weight_decay': 1e-6,
        'batch_size': 4,
        'num_workers': 8,
        'epochs': 60,
        'start': 0.5,
        'num_fold': N_FOLDS
    }

    # Run
    source = [str(i) for i in range(0, 10)]
    source += [chr(i) for i in range(97, 97 + 26)]
    source += [chr(i) for i in range(65, 65 + 26)]
    alphabet = ''.join(source)

    save_dir = '../kaggle_output/ocr_reg'
    store_path1 = os.path.join(save_dir, params['model1'])
    store_path2 = os.path.join(save_dir, params['model2'])
    print('Load from: ', store_path1, store_path2)

    # test
    root = '../kaggle/ocr_data/test_dataset/'
    images = glob.glob(os.path.join(root, '*.png'))
    name2idx = {}
    for i in range(len(images)):
        idx = int(images[i].split('/')[-1].split('.')[0])
        name2idx[idx] = images[i].split('/')[-1]
    name2idx = sorted(name2idx.items(), key=lambda item: item[0])
    images = []
    for i in name2idx:
        images.append(root + i[1])
    data_test = pd.DataFrame(index=range(len(images)))
    data_test['file'] = images

    test_dataset = TestData(data_test, transform=get_valid_transforms(params=params))
    test_data_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

    model_lists1 = []
    model_lists2 = []
    for fold in range(5):
        model = Net(params['model1'], 62 * 4, pretrained=False)
        model = model.to(params['device'])
        net_weights = torch.load(store_path1 + '/' + params['model1'] + "_fold{}.pth".format(fold + 1))
        model.load_state_dict(net_weights)
        model.eval()
        model_lists1.append(model)

        model = Net(params['model2'], 62 * 4, pretrained=False)
        model = model.to(params['device'])
        net_weights = torch.load(store_path2 + '/' + params['model2'] + "_fold{}.pth".format(fold + 1))
        model.load_state_dict(net_weights)
        model.eval()
        model_lists2.append(model)

    labels = []

    for img in tqdm(test_data_loader):
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()

        output = 0.
        for i in range(5):
            model = model_lists1[i]
            output += model(img)

        for i in range(5):
            model = model_lists2[i]
            output += model(img)

        output /= 10
        output = output.view(-1, 62)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        output = output.view(-1, 4)

        outputs = output.cpu().numpy()

        for val in outputs:
            label = ''.join([alphabet[i] for i in val])
            label = label.strip('.')
            labels.append(label)

    print(labels[:100])

    submit = pd.DataFrame(index=range(len(data_test)))
    submit['num'] = data_test['file']
    submit['num'] = submit['num'].apply(lambda x: x.split('.')[-2].split('/')[-1])
    submit['tag'] = labels
    print(submit)
    submit.to_csv(save_dir + '/' + params['model1'] + params['model2'] + '-avg.csv', index=None)
