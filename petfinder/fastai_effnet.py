from fastai.vision.all import *
from timm import create_model
from sklearn.model_selection import StratifiedKFold
import gc
import torch
from sklearn.metrics import mean_squared_error
import albumentations
from fastai.metrics import AccumMetric

set_seed(365, reproducible=True)
BATCH_SIZE = 32

seed = 365
set_seed(seed, reproducible=True)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

dataset_path = Path('./../../kaggle/petfinder-pawpularity-score/')
print(dataset_path.ls())

train_df = pd.read_csv(dataset_path / 'train.csv')

train_df['path'] = train_df['Id'].map(lambda x: str(dataset_path / 'train' / x) + '.jpg')
train_df = train_df.drop(columns=['Id'])
train_df = train_df.sample(frac=1).reset_index(drop=True)

len_df = len(train_df)
print(f"There are {len_df} images")

print(f"The mean Pawpularity score is {train_df['Pawpularity'].mean()}")
print(f"The median Pawpularity score is {train_df['Pawpularity'].median()}")
print(f"The standard deviation of the Pawpularity score is {train_df['Pawpularity'].std()}")

print(f"There are {len(train_df['Pawpularity'].unique())} unique values of Pawpularity score")

train_df['norm_score'] = train_df['Pawpularity']

num_bins = int(np.floor(1 + np.log2(len(train_df))))
train_df['bins'] = pd.cut(train_df['norm_score'], bins=num_bins, labels=False)
train_df['fold'] = -1
N_FOLDS = 10

strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['bins'])):
    train_df.iloc[train_index, -1] = i

train_df['fold'] = train_df['fold'].astype('int')


def petfinder_rmse(input, target):
    return torch.sqrt(F.mse_loss(input.flatten(), target))


class AlbumentationsTransform(RandTransform):
    "A transform handler for multiple `Albumentation` transforms"
    split_idx, order = None, 2

    def __init__(self, train_aug, valid_aug):
        store_attr()

    def before_call(self, b, split_idx):
        self.idx = split_idx

    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=np.array(img))['image']
        else:
            aug_img = self.valid_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


def get_train_aug(sz): return albumentations.Compose([
            albumentations.Resize(sz, sz),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
            albumentations.CoarseDropout(p=0.5),
])


def get_valid_aug(sz): return albumentations.Compose([
    albumentations.Resize(sz, sz)
], p=1.)


def get_data(fold):
    train_df_f = train_df.copy()
    # add is_valid for validation fold
    train_df_f['is_valid'] = (train_df_f['fold'] == fold)

    item_tfms = AlbumentationsTransform(get_train_aug(448), get_valid_aug(448))
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    dls = ImageDataLoaders.from_df(train_df_f,  # pass in train DataFrame
                                   valid_col='is_valid',  #
                                   seed=seed,  # seed
                                   fn_col='path',  # filename/path is in the second column of the DataFrame
                                   label_col='norm_score',  # label is in the first column of the DataFrame
                                   y_block=RegressionBlock,  # The type of target
                                   bs=BATCH_SIZE,  # pass in batch size
                                   num_workers=8,
                                   item_tfms=item_tfms,  # pass in item_tfms
                                   batch_tfms=batch_tfms)  # pass in batch_tfms

    return dls


# Valid Kfolder size
the_data = get_data(0)
assert (len(the_data.train) + len(the_data.valid)) == (len(train_df) // BATCH_SIZE)


def get_learner(fold_num):
    data = get_data(fold_num)

    model = create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=data.c)
    learn = Learner(data, model, loss_func=MSELossFlat(), metrics=AccumMetric(func=petfinder_rmse),
                    path='/home/wq/kaggle_output/tf_efficientnet_b4_ns').to_fp16()

    return learn, data


test_df = pd.read_csv(dataset_path / 'test.csv')
test_df['Pawpularity'] = [1] * len(test_df)
test_df['path'] = test_df['Id'].map(lambda x: str(dataset_path / 'test' / x) + '.jpg')
test_df = test_df.drop(columns=['Id'])
train_df['norm_score'] = train_df['Pawpularity']

all_preds = []
pred = []
target = []

for i in range(N_FOLDS):
    print(f'Fold {i} results')

    learn, data = get_learner(fold_num=i)

    learn.fit_flat_cos(20, 1e-4, pct_start=0., cbs=[SaveModelCallback(),
                                   EarlyStoppingCallback(monitor='petfinder_rmse', comp=np.less, patience=5)])

    # validate
    A = data.valid.dataset
    valid = [A[i][1] for i in range(len(A))]
    assert len(valid) == len(A)

    valid_dls = data.valid
    val_preds, _ = learn.get_preds(dl=valid_dls)

    pred.append(val_preds.to('cpu').numpy().reshape(1, -1).flatten())
    target.append(np.array([valid[i].item() for i in range(len(valid))]))

    learn = learn.to_fp32()
    learn.export(f'model_fold_{i}.pkl')

    del learn

    torch.cuda.empty_cache()

    gc.collect()

p = np.concatenate(pred) * 100
t = np.concatenate(target) * 100

rmse = round(mean_squared_error(t, p, squared=False), 3)
print('Efficientnet B4 ns CV: ', rmse)
