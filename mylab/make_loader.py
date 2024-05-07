import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import random
import math
from pathlib import Path
import glob
from skimage.io import imread

# バッチ単位で乱数をふる


class AlignedDataset(Dataset):
    def __init__(self, path) -> None:
        # データセットクラスの初期化

        directory_paths = [
            p for p in glob.glob(os.path.join(path, '*', '*')) if os.path.isdir(p)
        ]

        self.target_list = []
        for path in directory_paths:
            for _, _, fnames in sorted(os.walk(path)):
                target_file_path = os.path.join(
                    path, fnames[0]
                )
                self.target_list.append(target_file_path)

    def make_dataset(self, index) -> dict:
        # sourcery skip: inline-immediately-returned-variable

        target_path = self.target_list[index]
        target_numpy = imread(target_path)

        return target_numpy

    def __getitem__(self, index):
        return self.make_dataset(index)

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.target_list)
