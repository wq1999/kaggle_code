## **PetFinder.my - Pawpularity Contest**

### **Fastai**

#### **Augmentations**

```python
Brightness()
Contrast()
Hue()
Saturation()
```

#### Model

```
Swin-Transformer Lagre 224
Swin-Transformer Base 384
```

#### Train

```python
10 Folds
20 epoch
OneCycle

learn.fit_one_cycle(20, 1e-5, cbs=[SaveModelCallback(),EarlyStoppingCallback(monitor='petfinder_rmse', comp=np.less, patience=5)])
```

#### **TTA**

```python
# Fastai 
learner.tta(dl=test_dl, n=8, beta=0)
```

#### **Result**

|                                            |   CV   |    LB    |
| :----------------------------------------: | :----: | :------: |
| model1 swin_large_patch4_window7_224_in22k | 17.658 | 17.84621 |
| model2 swin_base_patch4_window12_384_in22k | 17.703 | 17.81121 |
|               model1+model2                | 17.435 | 17.77984 |
|     model1(lr=1e-5, CV: 17.606)+model2     | 17.414 | 17.77218 |

####  **Conclusion**

1. 相信线下CV分数，PB只有25%的数据，private leadboard有75%的数据，可能会过拟合PB。
2. 模型融合，增加模型多样性，因为时间限制，所有要选择合适的折数以及TTA的参数。
3. 单模10折也能取得比较好的分数。
