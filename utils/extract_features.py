import os
import librosa
from pyannote.database.util import load_rttm
from pyannote.core import Segment
import soundfile as sf
from tqdm import tqdm
import numpy as np
import pandas as pd
from resemblyzer import preprocess_wav, VoiceEncoder
from pyannote.audio import Inference, Model
import tempfile
import torch
import numpy as np
import sys

SAMPLE_RATE = 16000

def make_enrolls(path_rttm):
    enrolls = dict()
    new_enrolls = dict()
    speakers_files = dict()
    for _, _, filenames in os.walk(path_rttm):
        for filename in tqdm(filenames, position=0, file=sys.stdout):
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
                        wav = preprocess_wav(name_file)
                        emb_frame = encoder.embed_utterance(wav).tolist()
                        if speaker not in embeddings:
                            embeddings[speaker] = []
                        embeddings[speaker].append(emb_frame)

    return embeddings

def time2annot(time):
    time = round(time, 3)
    a = int(time // 0.655)
    b = round(time % 0.655, 2)
    idx = int(a * 64)
    idx += int((b * 100))

    return idx

def make_trainpart(root, path_corpus, path_rttm, enrolls, encoder, is_debug=False, stage='train'):

    # Create train dir
    datadir = os.path.join(root, 'custom_' + stage + '_pvad')
    idx_file = 0
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    audiodir = os.path.join(datadir, 'data')
    if not os.path.exists(audiodir):
        os.mkdir(audiodir)
    vectorsdir = os.path.join(datadir, 'd_vectors')
    if not os.path.exists(vectorsdir):
        os.mkdir(vectorsdir)

    # Create d-vectors model and init d-vectors list
    d_vectors = {}

    # Init utterance length, frame length and frame step
    utterance_length = 0.655
    frame_length = 0.025
    frame_step = 0.01

    for _, dirnames, _ in os.walk(path_corpus):
        for dirname in tqdm(dirnames, position=0, file=sys.stdout):

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
                        wav = preprocess_wav(name_file)
                        emb = encoder.embed_utterance(wav).tolist()
                        if speaker not in d_vectors:
                            d_vectors[speaker] = []
                        d_vectors[speaker].append(emb)

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
            for cnt in range(len(annot) // 64):

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
                    name_dir = stage + str(idx_file)
                    dir_utterance = os.path.join(audiodir, name_dir)
                    if not os.path.exists(dir_utterance):
                        os.mkdir(dir_utterance)
                    idx_file += 1

                    # Slicing audio
                    wav = audio[int(start): int(start + utterance_length*SAMPLE_RATE)].copy()

                    # Save audio
                    wav_file = os.path.join(dir_utterance, 'audio.wav')
                    sf.write(wav_file, wav, SAMPLE_RATE)

                    # Save annot
                    data = pd.Series(data)
                    annot_file = os.path.join(dir_utterance, 'annot.csv')
                    data.to_csv(annot_file, index=False)

                start += int(utterance_length*SAMPLE_RATE)
            if is_debug:
                break

    # Save d-vectors
    for speaker in d_vectors:
        df_vectors = pd.DataFrame(d_vectors[speaker])
        vectors_file = os.path.join(vectorsdir, speaker + '.csv')
        df_vectors.to_csv(vectors_file, index=False)

def prepare_eval_set(root, path_corpus, path_rttm, stage='test'):

    # Create train dir
    datadir = os.path.join(root, 'evaluate_annot_' + stage)
    idx_file = 0
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # Init utterance length, frame length and frame step
    utterance_length = 0.655
    frame_length = 0.025
    frame_step = 0.01

    for _, dirnames, _ in os.walk(path_corpus):
        for dirname in tqdm(dirnames, position=0, file=sys.stdout):

            # Skip not necessary content
            if dirname == 'audio':
                continue

            # Load rttm
            rttm_name = os.path.join(path_rttm, dirname + '.rttm')
            _, groundtruth = load_rttm(rttm_name).popitem()
            segments = list(groundtruth._tracks.keys())
            time_end = max([groundtruth._tracks.keys()[i].end for i in range(len(groundtruth._tracks.keys()))])

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
            annot = pd.DataFrame(annot)
            dir_to_annot = os.path.join(datadir, dirname)
            if not os.path.exists(dir_to_annot):
                os.mkdir(dir_to_annot)
            annot_file = os.path.join(dir_to_annot, 'annot.csv')
            annot.to_csv(annot_file, index=False)

def make_pairs_dataset(root, path_corpus_near, path_corpus_far, path_rttm, embeddings, 
                       pvad, is_debug=False, stage='train'):

    # Create train dir
    datadir = os.path.join(root, 'pairs_' + stage)
    idx_file = 0
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    audiodir = os.path.join(datadir, 'data')
    if not os.path.exists(audiodir):
        os.mkdir(audiodir)

    # Init utterance length, frame length and frame step
    utterance_length = 0.655
    frame_length = 0.025
    frame_step = 0.01

    pvad.eval()

    for _, dirnames, _ in os.walk(path_corpus_near):
        for dirname in tqdm(dirnames, position=0, file=sys.stdout):

            # Skip not necessary content
            if dirname == 'audio':
                continue

            # Load ami train part (near and far field)
            name_file = os.path.join(path_corpus_near, dirname, 'audio',
                                     dirname + '.Mix-Headset.wav')
            audio_near, _ = librosa.load(name_file, sr=SAMPLE_RATE)
            name_file = os.path.join(path_corpus_far, dirname, 'audio',
                                     dirname + '.Array1-01.wav')
            audio_far, _ = librosa.load(name_file, sr=SAMPLE_RATE)

            # Load rttm
            rttm_name = os.path.join(path_rttm, dirname + '.rttm')
            _, groundtruth = load_rttm(rttm_name).popitem()
            segments = list(groundtruth._tracks.keys())
            time_end = max([groundtruth._tracks.keys()[i].end for i in range(len(groundtruth._tracks.keys()))])

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
            for cnt in range(len(annot) // 64):

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
                    name_dir = stage + str(idx_file)
                    dir_utterance = os.path.join(audiodir, name_dir)
                    if not os.path.exists(dir_utterance):
                        os.mkdir(dir_utterance)
                    idx_file += 1

                    # Slicing audio
                    wav_near = audio_near[int(start): int(start + utterance_length*SAMPLE_RATE)].copy()
                    wav_near = wav_near.astype(np.float32, order='C') * 10
                    fbanks_near = librosa.feature.melspectrogram(y=wav_near, sr=SAMPLE_RATE,
                                                            n_fft=400,
                                                            hop_length=160,
                                                            n_mels=40).astype('float32').T[:-2]
                    logfbanks_near = torch.tensor(np.log10(fbanks_near + 1e-6))
                    
                    wav_far = audio_far[int(start): int(start + utterance_length*SAMPLE_RATE)].copy()
                    wav_far = wav_far.astype(np.float32, order='C') * 10
                    fbanks_far = librosa.feature.melspectrogram(y=wav_far, sr=SAMPLE_RATE,
                                                            n_fft=400,
                                                            hop_length=160,
                                                            n_mels=40).astype('float32').T[:-2]
                    logfbanks_far = torch.tensor(np.log10(fbanks_far + 1e-6))

                    # Choose target speaker
                    uniq_speakers_on_audio = set()
                    for speakers in data:
                        for speaker in speakers:
                            uniq_speakers_on_audio.add(speaker)
                    uniq_speakers_on_audio = list(uniq_speakers_on_audio)
                    if not uniq_speakers_on_audio:
                        target_speakers = np.random.choice(list(embeddings.keys()), size=2).tolist()
                    else:
                        target_speakers = [None] * 2
                        target_speakers[0] = np.random.choice(uniq_speakers_on_audio)
                        target_speakers[1] = np.random.choice(list(embeddings.keys()))
                    # Get feature maps and logits
                    with torch.no_grad():
                        for target_speaker in target_speakers:
                            emb = embeddings[target_speaker].sample().to_numpy()
                            emb = torch.as_tensor(emb).squeeze().repeat(64, 1)
                            stacked_input_near = torch.cat([logfbanks_near, emb], dim=-1)[None, ...].to(torch.float32)
                            stacked_input_far = torch.cat([logfbanks_far, emb], dim=-1)[None, ...].to(torch.float32)
                            output_near, logits_near = pvad.get_features(stacked_input_near)
                            output_far, logits_far = pvad.get_features(stacked_input_far)

                            # Save features
                            feature_file = os.path.join(dir_utterance, 'feature_map_near_' + str(target_speaker) + '.txt')
                            np.savetxt(feature_file, torch.round(output_near[0, ...], decimals=3))
                            feature_file = os.path.join(dir_utterance, 'feature_map_far_' + str(target_speaker) + '.txt')
                            np.savetxt(feature_file, torch.round(output_far[0, ...], decimals=3))
                            feature_file = os.path.join(dir_utterance, 'logits_near_' + str(target_speaker) + '.txt')
                            np.savetxt(feature_file, torch.round(logits_near[0, ...], decimals=3))
                            feature_file = os.path.join(dir_utterance, 'logits_far_' + str(target_speaker) + '.txt')
                            np.savetxt(feature_file, torch.round(logits_far[0, ...], decimals=3))

                    # Save annot
                    data = pd.Series(data)
                    annot_file = os.path.join(dir_utterance, 'annot.csv')
                    data.to_csv(annot_file, index=False)

                    # Save list of target speakers on frame
                    targets_file = os.path.join(dir_utterance, 'target_speakers.csv')
                    with open(targets_file, "w") as output:
                        output.write(str(target_speakers))
                    if is_debug:
                        return

                start += int(utterance_length*SAMPLE_RATE)
