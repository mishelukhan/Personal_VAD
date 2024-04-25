import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import ast
import librosa

SAMPLE_RATE = 16000

class AMIDataset(Dataset):
    """Dataset class for train and evaluation."""
    def __init__(self, stage, num_files, path_data, path_vectors, speakers):
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

        self.stage = stage
        self.num_files = num_files
        self.path_data = path_data
        self.vectors = {}
        for speaker in speakers:
            path_emb = os.path.join(path_vectors, speaker + '.csv')
            self.vectors[speaker] = pd.read_csv(path_emb)
        self.speakers = speakers


    def __len__(self):
        return self.num_files

    def __getitem__(self, idx: int):
        """Dataset method to extract data by index"""
        # Extract audio
        f_name = os.path.join(self.path_data, self.stage + str(idx), 'audio.wav')
        audio, _ = librosa.load(f_name, sr=SAMPLE_RATE)

        # Extract log mel filterbanks
        audio = audio.astype(np.float32, order='C') * 10
        fbanks = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE,
                                                n_fft=400,
                                                hop_length=160,
                                                n_mels=40).astype('float32').T[:-2]
        logfbanks = torch.tensor(np.log10(fbanks + 1e-6))

        # Extract annotation
        f_name = os.path.join(self.path_data, self.stage + str(idx), 'annot.csv')
        annot = pd.read_csv(f_name)
        annot = annot['0'].apply(ast.literal_eval).values

        # Choose target speaker
        uniq_speakers_on_audio = set()
        for speakers in annot:
            for speaker in speakers:
                uniq_speakers_on_audio.add(speaker)
        uniq_speakers_on_audio = list(uniq_speakers_on_audio)
        if not uniq_speakers_on_audio:
            target_speaker = np.random.choice(self.speakers)
        else:
            eps = np.random.rand()
            if eps > 0.5:
                target_speaker = np.random.choice(uniq_speakers_on_audio)
            else:
                target_speaker = np.random.choice(self.speakers)
        
        # Extract embedding
        emb = self.vectors[target_speaker].sample().to_numpy()
        emb = torch.as_tensor(emb).squeeze().repeat(64, 1)

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
        if logfbanks.shape[0] < 64:
            padding_value = (torch.ones(logfbanks.shape[1]) * (-10)).repeat(64 - logfbanks.shape[0], 1)
            logfbanks = torch.cat([logfbanks, padding_value], dim=0)
        stacked_input = torch.cat([logfbanks, emb], dim=-1)

        return stacked_input, labels
