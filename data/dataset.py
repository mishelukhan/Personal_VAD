import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import ast

class AMIDataset(Dataset):
    """Dataset class for train and evaluation."""
    def __init__(self, data, path_data, path_vectors):
        """Dataset class initializer.

        Args:
            data(list): List with folder names. Each folder should contains txt file
                        with Log-melfilterbanks energies and csv file with annotations.
            path_data (str): Path to directory with data
            path_vectors (str): Path to csv file with embeddings. Such csv should have structure like this one:
            name_speaker ||| embedding 
            Note that each speaker may have many embeddings extracted from different segments with audio.
            mode (str): If mode = train then 
            use_fc (bool, optional): Specifies, whether the model should use the
                last fully-connected hidden layer. Defaults to True.
            linear (bool, optional): Specifies the activation function used by the last
                hidden layer. If False, the tanh is used, if True, no activation is
                used. Defaults to False.
        """

        self.data = data
        self.path_data = path_data
        self.vectors = pd.read_csv(path_vectors)
        self.speakers = list(set(self.vectors['0']))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """Dataset method to extract data by index"""
        # Extract logmel filterbanks
        f_name = os.path.join(self.path_data, self.data[idx], 'filterbank.txt')
        logfbanks = np.loadtxt(f_name)
        logfbanks = torch.as_tensor(logfbanks)

        # Extract annotation
        f_name = os.path.join(self.path_data, self.data[idx], 'annot.csv')
        annot = pd.read_csv(f_name)
        eps = np.random.rand()
        if eps > 0.5:
            rand_ind = np.random.randint(0, len(annot))
            lst = ast.literal_eval(annot.iloc[rand_ind][0])
            if lst:
                random_speaker = np.random.choice(lst)
            else:
                random_speaker = np.random.choice(self.speakers)
        else:
            random_speaker = np.random.choice(self.speakers)

        # Extract embedding
        if random_speaker not in self.speakers:
            random_speaker = np.random.choice(self.speakers)
        emb = self.vectors[self.vectors['0'] == random_speaker].sample().drop(columns=['0']).to_numpy()
        emb = torch.as_tensor(emb).squeeze()
        new_annot = []
        for idx in range(len(annot)):
            lst = ast.literal_eval(annot.iloc[idx][0])
            if not lst:
                flag = 0
            elif random_speaker not in lst:
                flag = 1
            else:
                flag = 2
            new_annot.append(flag)
        new_annot = torch.as_tensor(new_annot, dtype=torch.long)
        input = torch.empty((logfbanks.shape[0], logfbanks.shape[1] + len(emb)))
        for idx in range(input.shape[0]):
            input[idx] = torch.cat((logfbanks[idx], torch.clone(emb)), dim=0)

        return input, new_annot