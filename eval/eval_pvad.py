import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import tempfile
import os
import librosa
import numpy as np
import soundfile as sf
from resemblyzer import preprocess_wav
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

SAMPLE_RATE = 16000

def eval_pvad(pvad, embedder, dataset, embeddings, device, 
              treshold_model=0.8, update_emb=False, treshold=0.1, averaging_factor = 0.9):
    frame = 0.655
    y_true = []
    y_pred = []
    y_probs = []
    softmax = nn.Softmax(dim=1)

    for file_name, amplitudes, annots in tqdm(dataset):
        start = 0
        for cnt in tqdm(range(len(annots) // 64)):
            audio = amplitudes[int(start*SAMPLE_RATE): int((start + frame)*SAMPLE_RATE)].copy()
            audio = audio.astype(np.float32, order='C') * 10
            fbanks = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=400,
                      hop_length=160, n_mels=40).astype('float32').T[:-2]
            logfbanks = torch.tensor(np.log10(fbanks + 1e-6), device=device)

            start += frame
            annot = annots[cnt * 64: (cnt + 1) * 64]
            for each in embeddings:
                if each not in dataset.get_speakers_files()[file_name]:
                    continue
                emb = torch.tensor(embeddings[each], device=device)
                input = torch.empty((logfbanks.shape[0], logfbanks.shape[1] + len(emb)))
                for idx in range(logfbanks.shape[0]):
                    input[idx] = torch.cat((logfbanks[idx], emb), dim=0)
                input = torch.unsqueeze(input, 0)
                with torch.no_grad():
                    output = pvad(input, [64])[0]
                probs = softmax(output[0])
                preds = probs[:, 2]

                if update_emb and preds.sum() > treshold:
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        name_file = os.path.join(tmpdirname, each + '.wav')
                        sf.write(name_file, audio, SAMPLE_RATE)
                        speech_prepr = preprocess_wav(name_file)
                        emb_frame = torch.tensor(embedder.embed_utterance(speech_prepr), device=device)
                    embeddings[each] = ((1 - averaging_factor) * emb_frame + averaging_factor * emb).tolist()

                predicts = []
                for idx in range(preds.shape[0]):
                    logits = preds[idx]
                    if logits.item() < treshold_model:
                        predicts.append(0)
                    else:
                        predicts.append(1)
                targets = []
                for gt in annot:
                    if (not gt) or each not in gt:
                        targets.append(0)
                    else:
                        targets.append(1)
                y_pred.extend(predicts)
                y_true.extend(targets)
                y_probs.extend(preds.detach().cpu().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    ap = average_precision_score(y_true, y_probs)
    map = ap.mean()
    roc_auc = roc_auc_score(y_true, y_probs)
    acc = accuracy_score(y_true, y_pred)
    params = [treshold, averaging_factor] if update_emb else None
    description = 'pvad' + ('_with_updating' if update_emb else '')

    return description, roc_auc, map, acc, params