import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import tempfile
import os
import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score

SAMPLE_RATE = 16000

def eval_embedder(embedder, dataset, embeddings, device, treshold_model=0.1, 
                  update_emb=False, treshold=0.3, averaging_factor = 0.9):
    frame = 0.655
    cos = nn.CosineSimilarity()
    y_true = []
    y_pred = []
    y_probs = []
    softmax = nn.Softmax(dim=1)

    for file_name, amplitudes, annots in tqdm(dataset):
        start = 0
        for cnt in tqdm(range(len(annots) // 64)):
            audio = amplitudes[int(start*SAMPLE_RATE): int((start + frame)*SAMPLE_RATE)].copy()
            audio = audio.astype(np.float32, order='C')
            with tempfile.TemporaryDirectory() as tmpdirname:
                name_file = os.path.join(tmpdirname, '1.wav')
                sf.write(name_file, audio, SAMPLE_RATE)
                emb_frame = embedder(name_file).tolist()

            start += frame
            annot = annots[cnt * 64: (cnt + 1) * 64]
            for each in embeddings:
                if each not in dataset.get_speakers_files()[file_name]:
                    continue
                emb_1 = torch.unsqueeze(torch.tensor(emb_frame, device=device), 0)
                emb_2 = torch.tensor(embeddings[each], device=device)
                pred = cos(emb_1, emb_2)

                y_probs.append(pred.item())

                if update_emb and pred.item() > treshold:
                    embeddings[each] = ((1 - averaging_factor) * emb_1 + averaging_factor * emb_2).tolist()
                targets = []
                for gt in annot:
                    if (not gt) or (each not in gt):
                        targets.append(0)
                    else:
                        targets.append(1)
                y_true.append(int((sum(targets) / len(targets)) > 0.5))

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = np.array([int(el > treshold_model) for el in y_probs])
    acc = accuracy_score(y_true, y_pred)

    ap = average_precision_score(y_true, y_probs)
    map = ap.mean()
    roc_auc = roc_auc_score(y_true, y_probs)
    params = [treshold, averaging_factor] if update_emb else None
    description = 'pvad_with_embedder' + ('_with_updating' if update_emb else '')

    return description, roc_auc, map, acc, params
        
        
