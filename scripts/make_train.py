import os
import librosa
from pyannote.database.util import load_rttm
from pyannote.core import Segment
import soundfile as sf
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from resemblyzer import preprocess_wav, VoiceEncoder
from pyannote.audio import Inference, Model
import tempfile
import torch
import numpy as np

SAMPLE_RATE = 16000

def make_enrolls(path_rttm):
    enrolls = dict()
    new_enrolls = dict()
    speakers_files = dict()
    for _, _, filenames in os.walk(path_rttm):
        for filename in tqdm(filenames):
            enrolls[filename[:-5]] = dict()
            # Load files
            rttm_name = os.path.join(path_rttm, filename)
            _, groundtruth = load_rttm(rttm_name).popitem()
            segments = list(groundtruth._tracks.keys())
            upper_time = groundtruth._tracks.keys()[-1].end

            # Find all speakers
            speakers_session = set()
            speakers_enrolls = set()
            for seg in segments:
                speaker = list(groundtruth._tracks[seg].values())[0]
                speakers_session.add(speaker)

            # Init window
            start = 0
            win_len = 5
            shift_window = 1
            window = Segment(start, start + win_len)
            intersect = []
            for seg in segments:
                if seg.intersects(window):
                    speaker = list(groundtruth._tracks[seg].values())[0]
                    intersect.append((speaker, filename[:-5], seg))

            # Iterate over groundtruth annot
            while window.end < upper_time:

                # Check if there is only one speaker in window
                if len(intersect) == 1:
                    speaker = intersect[0][0]
                    speakers_enrolls.add(speaker)
                    if speaker not in enrolls[filename[:-5]]:
                        enrolls[filename[:-5]][speaker] = list()
                    name_f = intersect[0][1]
                    seg = intersect[0][2]
                    if seg.end - seg.start > 4:
                        new_start = max(seg.start, window.start)
                        new_end = min(seg.end, window.end)
                        new_seg = Segment(new_start, new_end)
                        enrolls[filename[:-5]][speaker].append(new_seg)

                # Shift window
                window = Segment(window.start + shift_window, window.end + shift_window)
                intersect = []
                for seg in segments:
                    if seg.intersects(window):
                        speaker = list(groundtruth._tracks[seg].values())[0]
                        intersect.append((speaker, filename[:-5], seg))
            new_enrolls[filename[:-5]] = dict()
            for speak in enrolls[filename[:-5]]:
                for seg in enrolls[filename[:-5]][speak]:
                    if seg.end - seg.start > 4:
                        if speak not in new_enrolls[filename[:-5]]:
                            new_enrolls[filename[:-5]][speak] = []
                        new_enrolls[filename[:-5]][speak].append(seg)

    return new_enrolls



def make_embs(path_corpus, enrolls, encoder):
    # Make dict with embs
    embeddings = dict()
    emb_file = []
    for _, dirnames, _ in os.walk(path_corpus):
        for dirname in tqdm(dirnames):

            # Skip not necessary content
            if dirname == 'audio':
                continue
            name_file = os.path.join(path_corpus, dirname, 'audio',
                                     dirname + '.Mix-Headset.wav')
            audio, _ = librosa.load(name_file, sr=SAMPLE_RATE)

            enroll_file = enrolls[dirname]
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Create temporary directory for correct work of VoiceEncoder
                for speaker in tqdm(enroll_file):
                    for seg in enroll_file[speaker]:
                        speech = audio[int(seg.start*SAMPLE_RATE): int(seg.end*SAMPLE_RATE)].copy()
                        name_file = os.path.join(tmpdirname, speaker + '.wav')
                        sf.write(name_file, speech, SAMPLE_RATE)
                        emb_frame = encoder(name_file).tolist()
                        if speaker not in embeddings:
                            embeddings[speaker] = []
                        embeddings[speaker].append(emb_frame)

    return embeddings

def time2annot(time):
    a = time // 0.655
    b = time % 0.655
    idx = int(a * 64)
    if b < 0.025:
        return idx
    idx += (b - 0.025) // 0.01 + 1

    return int(idx)

def make_trainpart(root, path_corpus, path_rttm, enrolls, embedder, device):

    # Create train dir
    datadir = os.path.join(root, 'custom_train_pvad')
    idx_file = 0
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    audiodir = os.path.join(datadir, 'data')
    if not os.path.exists(audiodir):
        os.mkdir(audiodir)

    # Create d-vectors model and init d-vectors list
    d_vectors = []

    # Init utterance length, frame length and frame step
    utterance_length = 0.655
    frame_length = 0.025
    frame_step = 0.01

    for _, dirnames, _ in os.walk(path_corpus):
        for dirname in tqdm(dirnames):

            # Skip not necessary content
            if dirname == 'audio':
                continue

            # Load ami train part
            name_file = os.path.join(path_corpus, dirname, 'audio',
                                     dirname + '.Mix-Headset.wav')
            audio, _ = librosa.load(name_file, sr=SAMPLE_RATE)

            # Load rttm
            rttm_name = os.path.join(path_rttm, dirname + '.rttm')
            _, groundtruth = load_rttm(rttm_name).popitem()
            segments = list(groundtruth._tracks.keys())
            time_end = max([groundtruth._tracks.keys()[i].end for i in range(len(groundtruth._tracks.keys()))])

            # Make d-vectors
            enroll_file = enrolls[dirname]
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Create temporary directory for correct work of VoiceEncoder
                for speaker in enroll_file:
                    for seg in enroll_file[speaker]:
                        speech = audio[int(seg.start*SAMPLE_RATE): int(seg.end*SAMPLE_RATE)].copy()
                        name_file = os.path.join(tmpdirname, speaker + '.wav')
                        sf.write(name_file, speech, SAMPLE_RATE)
                        emb = embedder(name_file).tolist()
                        d_vectors.append((speaker, *emb))

            # Make annot
            size_annot = time2annot(time_end) + 1
            annot = [[] for _ in range(int(size_annot))]
            for seg in segments:
                speaker = list(groundtruth._tracks[seg].values())[0]
                start = seg.start
                end = seg.end
                idx_start = time2annot(start)
                idx_end = time2annot(end)
                for idx in range(idx_start, idx_end + 1):
                    try:
                        annot[idx].append(speaker)
                    except IndexError:
                        print(time_end, size_annot, idx, idx_start, idx_end)
                        break

            start = 0
            for cnt in tqdm(range(len(annot) // 64)):

                # Choose sublist in annot according to cnt
                data = annot[cnt*64: (cnt+1)*64]
                is_full = False
                for frame in data:
                    if frame:
                        is_full = True
                        break
                eps = np.random.rand()
                if is_full or eps > 0.8:

                    # Create train file directory
                    name_dir = 'train' + str(idx_file)
                    dir_utterance = os.path.join(audiodir, name_dir)
                    if not os.path.exists(dir_utterance):
                        os.mkdir(dir_utterance)
                    idx_file += 1

                    # Make log melfilterbanks
                    wav_file = audio[int(start): int(start + utterance_length*SAMPLE_RATE)].copy()
                    wav_file = wav_file.astype(np.float32, order='C') * 10
                    fbanks = librosa.feature.melspectrogram(y=wav_file, sr=SAMPLE_RATE,
                                                            n_fft=400,
                                                            hop_length=160,
                                                            n_mels=128).astype('float32').T[:-2]
                    logfbanks = torch.tensor(np.log10(fbanks + 1e-6), device=device)

                    # Save log melfilterbanks
                    mel_file = os.path.join(dir_utterance, 'filterbank.txt')
                    np.savetxt(mel_file, logfbanks)

                    # Save annot
                    data = pd.Series(data)
                    annot_file = os.path.join(dir_utterance, 'annot.csv')
                    data.to_csv(annot_file, index=False)

                start += utterance_length*SAMPLE_RATE


    # Save d-vectors
    d_vectors = pd.DataFrame(d_vectors)
    vectors_file = os.path.join(datadir, 'd_vectors.csv')
    d_vectors.to_csv(vectors_file, index=False)
    