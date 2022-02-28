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
import time

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


# inference
def predict(weight_files, store_path):
    # test dataset
    root = '/home/wq/kaggle/ocr_data/test_dataset/'
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
    print(data_test)

    test_dataset = TestData(data_test, transform=get_valid_transforms(params=params))
    test_data_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

    # model
    model_lists = []
    for weight_file in weight_files:
        print(f"filename:{weight_file}")

        model = Net(params['model'], 62 * 4, pretrained=False)
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
            output = output.view(-1, 62)
            output = nn.functional.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)
            output = output.view(-1, 4)
            output = output.cpu().numpy()

            for i in range(len(output)):
                pred = ''.join([alphabet[idx] for idx in output[i]])
                result.append([filename[i].split('.')[-2].split('/')[-1], pred])

        result = pd.DataFrame(np.array(result), columns=["num", "tag"])
        ttar.append(result)

    r = ttar[0]

    tm = str(time.strftime('%m%d%H%M'))
    wc = 'tf_efficientnet_b5_ns_5folds'
    r.to_csv(store_path + "/submit_{}_{}.csv".format(wc, tm), index=False)


if __name__ == '__main__':
    # CFG
    N_FOLDS = 5
    params = {
        'model': 'tf_efficientnet_b5_ns',
        'im_size': 160,
        'device': device,
        'lr': 1e-3,
        'use_amp': True,
        'weight_decay': 1e-6,
        'batch_size': 32,
        'num_workers': 8,
    }

    # Run
    source = [str(i) for i in range(0, 10)]
    source += [chr(i) for i in range(97, 97 + 26)]
    source += [chr(i) for i in range(65, 65 + 26)]
    alphabet = ''.join(source)

    save_dir = './'
    store_path = os.path.join(save_dir, params['model'])
    print('Load from: ', store_path)

    predict(glob.glob(store_path + "/*.pth"), store_path)
