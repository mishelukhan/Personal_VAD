import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import tempfile
import os
import librosa
import numpy as np
import soundfile as sf
from resemblyzer import preprocess_wav
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, confusion_matrix

SAMPLE_RATE = 16000

def eval_pvad(pvad, encoder, dataset, embeddings, device,
              update_emb=True, treshold=0.9, averaging_factor=0.9):
    frame = 0.655
    softmax = nn.Softmax(dim=2)
    used_embeddings = {}
    for speaker in dataset.speakers:
        # used_embeddings[speaker] = embeddings[speaker].sample().to_numpy()
        used_embeddings[speaker] = embeddings[speaker].iloc[0].to_numpy()
    accuracies = []
    f1_scores = []
    map_scores = []
    examples_audio = []
    pvad.eval()
    for amplitudes, annots, unique_speakers in tqdm(dataset):
        start = 0
        y_true = []
        y_pred = []
        y_probs = []
        for cnt in tqdm(range(len(annots) // 64)):
            audio = amplitudes[int(start*SAMPLE_RATE): int((start + frame)*SAMPLE_RATE)].copy()
            audio = audio.astype(np.float32, order='C') * 10
            fbanks = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=400,
                      hop_length=160, n_mels=40).astype('float32').T[:-2]
            logfbanks = torch.tensor(np.log10(fbanks + 1e-6), device=device)

            start += frame
            annot = annots[cnt * 64: (cnt + 1) * 64]
            accuracy_frame = []
            y_true_frame = []
            y_pred_frame = []
            y_probs_frame = []
            for each in unique_speakers:
                # Form labels
                labels = []
                for item in annot:
                    if not item:
                        flag = 0
                    elif each not in item:
                        flag = 1
                    else:
                        flag = 2
                    labels.append(flag)
                labels = torch.as_tensor(labels, dtype=torch.long)
                emb = used_embeddings[each]
                emb = torch.as_tensor(emb).to(device).squeeze().repeat(64, 1)
                stacked_input = torch.cat([logfbanks, emb], dim=-1)[None, ...].to(torch.float32)
                with torch.no_grad():
                    output, _ = pvad(stacked_input)
                probs = np.squeeze(softmax(output).detach().cpu().numpy(), axis=0)
                new_probs = np.transpose(np.vstack([probs[:, 0] + probs[:, 1], probs[:, 2]]))
                prediction = new_probs.argmax(axis=1)
                targets = labels.int().detach().cpu().numpy().flatten()
                targets = (targets > 1).astype(int)
    
                if update_emb and new_probs[:, 1].mean() > treshold:
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        name_file = os.path.join(tmpdirname, each + '.wav')
                        sf.write(name_file, audio, SAMPLE_RATE)
                        speech_prepr = preprocess_wav(name_file)
                        emb_frame = torch.tensor(encoder.embed_utterance(speech_prepr), device=device)
                    used_embeddings[each] = ((1 - averaging_factor) * emb_frame + averaging_factor * used_embeddings[each])

                acc = accuracy_score(targets, prediction)
                accuracy_frame.append(acc)

                y_true_frame.append(targets)
                y_pred_frame.append(prediction)
                y_probs_frame.append(new_probs)

            mean_accuracy_frame = sum(accuracy_frame) / len(accuracy_frame)
            accuracies.append(mean_accuracy_frame)
            y_true.extend(y_true_frame)
            y_pred.extend(y_pred_frame)
            y_probs.extend(y_probs_frame)

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_probs = np.array(y_probs)[:, :, 1].flatten()
        
        f1_res = f1_score(y_true, y_pred)
        map_res = average_precision_score(y_true, y_probs)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f'Mean accuracy: {sum(accuracies) / len(accuracies)}')
        print(f'Mean f1-score: {f1_res}')
        print(f'Mean average precision: {map_res}')
        print(f'False Acceptance Rate: {fn / (tp + fn)}')
        print(f'False Rejection Rate: {fp / (tn + fp)}')


def evaluate_target_oversuppresion_model(model, loader, device, field='near'):
    model.eval()
    with torch.no_grad():
        accuracies_val = []
        f1_val = []
        map_val = []
        y_true = []
        y_pred = []
        for iteration, batch_features in enumerate(tqdm(loader)):
            input_map = batch_features[0].to(device).to(torch.float32)
            logits = batch_features[1].to(device)
            target_map = batch_features[2].to(device).flatten()
            output_map, _ = model(input_map)
            logits[..., 2] = logits[..., 2] + output_map[..., 0]

            prediction = logits.argmax(dim=2).int().detach().cpu().numpy().flatten()
            probs = logits.detach().cpu().flatten(start_dim=0, end_dim=-2).numpy()
            targets = target_map.detach().cpu().numpy().flatten()
            targets_oh = np.eye(3)[targets]

            acc = accuracy_score(targets, prediction)

            f1 = f1_score(targets, prediction, average='micro')

            map_score = average_precision_score(targets_oh, probs, average='micro')
            accuracies_val.append(acc)
            f1_val.append(f1)
            map_val.append(map_score)

            y_true.append(targets)
            y_pred.append(prediction)
        y_probs = np.array(y_probs)[:, :, 1].flatten()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f'Test accuracy: {sum(accuracies_val) / len(accuracies_val)}')
        print(f'Test f1-score: {sum(f1_val) / len(f1_val)}')
        print(f'Test map-score: {sum(map_val) / len(map_val)}')
        print(f'False Acceptance Rate: {fn / (tp + fn)}')
        print(f'False Rejection Rate: {fp / (tn + fp)}')
