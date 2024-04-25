import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import ast
import librosa

class DatasetExactField(Dataset):
    """Dataset class for train and evaluation."""
    def __init__(self, data, path_data, field):
        """Dataset class initializer.

        Args:
            data(list): List with folder names. Each folder should contains audio file
                        and csv file with annotations.
            path_data (str): Path to directory with data
        """
        self.data = data
        self.path_data = path_data
        self.field = field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """Dataset method to extract data by index"""
        f_name = os.path.join(self.path_data, self.data[idx], 'target_speakers.csv')
        target_speakers = list(pd.read_csv(f_name).columns)
        target_speakers[0] = target_speakers[0][2:-1]
        target_speakers[1] = target_speakers[1][2:-2]
        
        # Randomly select speaker as target
        target_speaker = np.random.choice(target_speakers)

        # Extract annotation
        f_name = os.path.join(self.path_data, self.data[idx], 'annot.csv')
        annot = pd.read_csv(f_name)
        annot = annot['0'].apply(ast.literal_eval).values

        if self.field == "near":
            # Extract feature map from near field
            f_name = os.path.join(self.path_data, self.data[idx], 'feature_map_near_' + target_speaker + '.txt')
            fmap = np.loadtxt(f_name)
            fmap = torch.from_numpy(fmap)
            f_name = os.path.join(self.path_data, self.data[idx], 'logits_near_' + target_speaker + '.txt')
            logits = np.loadtxt(f_name)
            logits = torch.from_numpy(logits)
        else:
            # Extract feature map from far field
            f_name = os.path.join(self.path_data, self.data[idx], 'feature_map_far_' + target_speaker + '.txt')
            fmap = np.loadtxt(f_name)
            fmap = torch.from_numpy(fmap)
            f_name = os.path.join(self.path_data, self.data[idx], 'logits_far_' + target_speaker + '.txt')
            logits = np.loadtxt(f_name)
            logits = torch.from_numpy(logits)

        # Form labels
        labels = []
        for item in annot:
            if not item:
                flag = 0
            elif target_speaker not in item:
                flag = 1
            else:
                flag = 2
            labels.append(flag)
        labels = torch.as_tensor(labels, dtype=torch.long)

        return fmap, logits, labels
