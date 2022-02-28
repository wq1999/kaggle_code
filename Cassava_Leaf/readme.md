## **Cassava Leaf Disease Classification**

### **model**

```
CNN: tf_efficientnet_b4_ns size=512
Transformer: Swin Transformer size=384
```

### **loss**

```
Focal Loss

```

### **aug**

```
train:
    albumentations.RandomResizedCrop(DIM, DIM),
    albumentations.Transpose(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),
    albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    albumentations.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    albumentations.CoarseDropout(p=0.5),
    albumentations.Cutout(p=0.5),
valid:
	albumentations.CenterCrop(DIM, DIM, p=1.),
    albumentations.Resize(DIM, DIM),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),

```

### **train**

```
KFOLD: 5
Optimizer: Adam
LR: 1e-4
Scheduler: CosineAnnealingWarmRestarts
```

### **result**

```
effnet-b4 v1:  size=512, bs=32, epoch=10, lr=1e-4, scheduler=epoch update, T_0=10, loss=focal_loss, AdamW, SEED=999, TTA=8  CV: 0.8924 LB: 0.8967

swin-t-384 v1:  size=384, bs=32, epoch=10, lr=1e-4, scheduler=epoch update, T_0=10, loss=focal_loss, AdamW, SEED=999, TTA=8  CV: 0.893 LB: 0.8999

effnet-b4 v1 + swin-t-384 v1: LB: 0.9007(TTA=5)

```

