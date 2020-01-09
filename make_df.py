import numpy as np
import pandas as pd
import os
import cv2
from utils import make_mask_
from fastai.vision import *
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
TRAIN = './images/'
MASKS = './masks/'
TRAIN_N = './images_n/'
HARD_NEGATIVE = '../input/pred.csv'


def mask2enc(mask, shape=(1600, 256), n=4):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append('')
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs

def open_mask(fn:PathOrStr, div:bool=True, convert_mode:str='L', cls:type=ImageSegment,
        after_open:Callable=None)->ImageSegment:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        #generate empty mask if file doesn't exist
        x = PIL.Image.open(fn).convert(convert_mode) \
          if Path(fn).exists() \
          else PIL.Image.fromarray(np.zeros((256, 224)).astype(np.uint8))
    if after_open: x = after_open(x)
    x = pil2tensor(x,np.float32)
    return cls(x)


if __name__ == '__main__':
    df = pd.read_csv('./cropped_df.csv')
    df = df.fillna('')
    print(df.columns)
    df['defects'] = (df['1'] != '').astype(int)+(df['2'] != '').astype(int)+(df['3'] != '').astype(int)+(df['4'] != '').astype(int)
    print(df['defects'].value_counts())
    # preds = pd.read_csv('../input/pred.csv')
    # img_neg = preds['fname'].unique()[:15000]
    # print(len(img_neg))
    kf = GroupShuffleSplit(n_splits=5, random_state=2019)
    for train_idx, val_idx in kf.split(df, groups=df['Image']):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        break
    print(train_df.head())
    print(val_df.head())
    print(train_df['defects'].value_counts())
    print(val_df['defects'].value_counts())
    q = [['a']] * 10
    print(len(q))
    print(q)
    '''
    a = '640b3aede_1.png'
    mask = cv2.imread('./masks/'+a)
    print(type(mask))
    rles = mask2enc(mask)
    df = pd.DataFrame([rles], columns=[1,2,3,4])
    df['ImageId'] = '640b3aede_1.png'
    # print(df.iloc[0]["ImageId"])
    print('ok')
    print(rles)
    image_id, mask = make_mask_(0, df)
    print(type(mask.sum()))

    print('mask sum %d' %(mask.sum()))
    print(mask.shape)
    img = cv2.imread('./images/'+a)
    print(img.shape)
    fig, ax = plt.subplots(figsize=(15, 15))
    # ax.imshow(img)
    # ax.imshow(mask)
    palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]
    mask = mask.astype('uint8')
    print(mask[:, :, 0].astype('int').sum())
    print(mask[:, :, 1].astype('int').sum())
    print(mask[:, :, 2].astype('int').sum())
    print(mask[:, :, 3].astype('int').sum())
    for ch in range(3):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    ax.imshow(img)
    plt.show()
    '''
    '''
    # Path(TRAIN).lstat()
    # img = cv2.imread('./masks/0002cc93b_3.png')
    # print(img.shape)
    # input()
    img_p = set([p.stem[:-2] for p in Path(TRAIN).ls()])
    # print(img_p)
    neg = list(pd.read_csv(HARD_NEGATIVE).head(12000).fname)
    neg = [Path(TRAIN_N) / f for f in neg]

    img_n = set([p.stem for p in neg])
    # print(neg)
    # print(img_n)
    img_set = img_p | img_n
    img_p_list = sorted(img_p)
    img_n_list = sorted(img_n)
    img_list = img_p_list + img_n_list
    print(img_list)
    print(type(img_list))
    input()
    lis = os.listdir('./images')
    df = pd.DataFrame()
    df['ImageId'] = lis
    print(df['ImageId'].head())
    print(len(lis))'''