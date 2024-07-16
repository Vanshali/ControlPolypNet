import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import os


class MyTestDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./sample_test.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./polyp_data/' + source_filename)  # replace directory "polyp_data" with your test data source directory
        target = cv2.imread('./polyp_data/' + target_filename)
        
        #source = cv2.resize(source, (64, 64))   # ！！！！！！！！！
        #target = cv2.resize(target, (64, 64))     # ！！！！！！！！！


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float16) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float16) / 127.5) - 1.0
        

        return dict(jpg=target, txt=prompt, hint=source, target_name = target_filename)
