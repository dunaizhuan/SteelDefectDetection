from torch.utils.data import DataLoader, Dataset
import cv2
import os
from utils import make_mask,mask2enc,make_mask_
import numpy as np
import pandas as pd
from albumentations import (HorizontalFlip, Normalize, Compose, Resize, RandomRotate90, Flip, RandomCrop, PadIfNeeded)
from albumentations.pytorch import ToTensor
from sklearn.model_selection import train_test_split,GroupKFold,KFold,GroupShuffleSplit
path = './input/'
RANDOM_STATE = 2019


class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images",  image_id)
        # img = Image.open(image_path)
        # img = np.array(img)[:, :, 0]
        img = cv2.imread(image_path)[:, :, 0]
        img = img[:, :, np.newaxis]
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)


class SteelDatasetCopped(Dataset):
    def __init__(self, df, data_folder, mean= (0.41009), std= (0.16991), phase='train'):
        self.df = df
        self.root = data_folder
        self.mean = (0.3959)
        self.std = (0.1729)
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask = make_mask_(idx, self.df)
        # print(image_id)
        image_path = os.path.join(self.root, "images", image_id)
        try:
            img = cv2.imread(image_path)[:, :, 0]
        except Exception:
            image_path = os.path.join(self.root, "images_n", image_id)
            img = cv2.imread(image_path)[:, :, 0]
        img = img[:, :, np.newaxis]
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":

        list_transforms.extend(
            [
                # PadIfNeeded(min_height=256, min_width=256),
                # RandomCrop(height=256, width=256, p=1),
                # RandomCrop(height=224, width=224, p=1),
                HorizontalFlip(p=0.5),  # only horizontal flip as of now
                Flip(p=0.5),
                # RandomRotate90(p=0.5),
                # PadIfNeeded(min_height=256, min_width=256)
            ]
        )
    else:
        list_transforms.extend(
            [
                PadIfNeeded(min_height=256, min_width=256),
            ]
        )

    list_transforms.extend(
        [
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms


def provider(
        data_folder,
        df_path,
        phase,
        mean=None,
        std=None,
        batch_size=4,
        num_workers=4,
        cropped=False
):
    '''Returns dataloader for the model training'''
    if cropped ==False:
        df = pd.read_csv(df_path)
        # some preprocessing
        # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
        df['defects'] = df.count(axis=1)
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=RANDOM_STATE)
        df = train_df if phase == "train" else val_df
        image_dataset = SteelDataset(df, data_folder, mean, std, phase)
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
         )
    else:
        if os.path.exists('./other_thing/cropped_df.csv'):
            df_ = pd.read_csv('./other_thing/cropped_df.csv')
            df_ = df_.fillna('')
        else:
            print('Prepare rle ing')
            df = pd.DataFrame()
            df['ImageId'] = os.listdir('./other_thing/images')
            df['Image'] = df['ImageId'].apply(lambda x: x.split('.')[0][:-2])
            predictions = []
            for imgid in os.listdir('./other_thing/images'):
                mask = cv2.imread('./other_thing/masks/'+imgid)
                rles = mask2enc(mask)
                predictions.append(rles)

            img_neg = pd.read_csv('./input/pred.csv')
            img_neg = img_neg['fname'].unique()[:15000]
            df2 = pd.DataFrame()
            df2['ImageId'] = img_neg
            df2['Image'] = df2['ImageId'].apply(lambda x: x.split('.')[0][:-2])
            predictions2 = [['', '', '', '']]*15000

            df_ = pd.DataFrame(predictions2+predictions, columns=[1, 2, 3, 4])
            df_['ImageId'] = pd.concat([df2, df], axis=0)['ImageId'].values
            df_['Image'] = pd.concat([df2, df], axis=0)["Image"].values
            df_.to_csv('./other_thing/cropped_df.csv', index=False)
            print('finish prepare rle!')

        kf = GroupShuffleSplit(n_splits=5, random_state=RANDOM_STATE)
        for train_idx, val_idx in kf.split(df_, groups=df_['Image']):
            train_df = df_.iloc[train_idx]
            val_df = df_.iloc[val_idx]
            break
        df_ = train_df if phase == "train" else val_df

        image_dataset = SteelDatasetCopped(df_, './other_thing/', phase=phase)
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
         )
    return dataloader


if __name__ == '__main__':

    '''
    print('Prepare rle ing')
    df = pd.DataFrame()
    df['ImageId'] = os.listdir('./other_thing/images')
    df['Image'] = df['ImageId'].apply(lambda x: x.split('.')[0][:-2])
    predictions = []
    for imgid in df['ImageId'].unique():
        mask = cv2.imread('./other_thing/images/' + imgid)
        rles = mask2enc(mask)
        predictions.append(rles)

    df_ = pd.DataFrame(predictions, columns=['1', '2', '3', '4'])
    print('finished !')
    df_['ImageId'] = df['ImageId']
    df_['Image'] = df['Image']
    df_['defects'] = (df_['1']!='').astype(int)+(df_['2']!='').astype(int)+(df_['3']!='').astype(int)+(df_['4']!='').astype(int)
    print(df_['defects'].value_counts(dropna=False))
    print(df[df_['ImageId']=='19aef9172_0'])
    '''
