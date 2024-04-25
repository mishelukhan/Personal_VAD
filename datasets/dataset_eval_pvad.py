import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import ast
import librosa

SAMPLE_RATE = 16000

class EvalDataset(Dataset):
    """Dataset class for train and evaluation."""
    def __init__(self, path_data, path_annot, path_vectors, mode='near'):
        """Dataset class initializer.

        Args:
            data(list): List with folder names. Each folder should contains audio file
                        and csv file with annotations.
            path_data (str): Path to directory with data
            path_vectors (str): Path to csv file with embeddings. Such csv should have structure like this one:
            name_speaker ||| embedding 
            Note that each speaker may have many embeddings extracted from different segments in audio.
            speakers (set): Set of all speakers in corpus
        """

        self.path_data = path_data
        self.path_annot = path_annot
        self.vectors = {}
        self.speakers = [name[:-4] for name in next(os.walk(path_vectors))[2]]
        for speaker in self.speakers:
            path_emb = os.path.join(path_vectors, speaker + '.csv')
            self.vectors[speaker] = pd.read_csv(path_emb)
        self.dirnames = next(os.walk(path_data))[1]
        self.mode = mode


    def __len__(self):
        return len(self.dirnames)

    def __getitem__(self, idx: int):
        """Dataset method to extract data by index"""
        # Extract audio
        if self.mode == 'near':
            f_name = os.path.join(self.path_data, self.dirnames[idx], 'audio', self.dirnames[idx] + '.Mix-Headset.wav')
        elif self.mode == 'far':
            f_name = os.path.join(self.path_data, self.dirnames[idx], 'audio', self.dirnames[idx] + '.Array1-01.wav')
        else:
            raise Exception("Try another field")
        audio, _ = librosa.load(f_name, sr=SAMPLE_RATE)

        # Extract annotation
        f_name = os.path.join(self.path_annot, self.dirnames[idx], 'annot.csv')
        annot = pd.read_csv(f_name)
        labels = []
        for idx in range(annot.shape[0]):
            labels_frame = []
            for speaker in annot.iloc[idx].values.tolist():
                if isinstance(speaker, str):
                    labels_frame.append(speaker)
            labels.append(labels_frame)
        unique_speakers = []
        for idx in range(annot.shape[1]):
            unique_speakers.extend(annot[str(idx)].unique().astype(str).tolist())
        unique_speakers = set(unique_speakers)
        unique_speakers.remove('nan')
        unique_speakers = list(unique_speakers)

        return audio, labels, unique_speakers