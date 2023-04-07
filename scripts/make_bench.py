import sys
import os
from pyannote.database.util import load_rttm
import librosa
import torch
from pyannote.core import Segment
import numpy as np
import soundfile as sf

class AMITest(torch.utils.data.Dataset):
    def __init__(self, data, ground_truth):
        self.data = data
        self.ground_truth = ground_truth
        self.file_list = []
        for dirpath, dirnames, filenames in os.walk(self.data):
            self.file_list = dirnames
            break
        
    def __len__(self):
        return len(self.file_list)
    
    def get_file_list(self):
        return self.file_list

    def __getitem__(self, idx: int):
        sound_name = os.path.join(self.data, self.file_list[idx], 'audio', 
                                  self.file_list[idx] + '.Mix-Headset.wav')
        amplitudes, _ = librosa.load(sound_name, sr=SAMPLE_RATE)
        rttm_name = os.path.join(self.ground_truth, self.file_list[idx] + '.rttm')
        _, groundtruth = load_rttm(rttm_name).popitem()

        return amplitudes, groundtruth


def get_enroll(dataset):
    enroll_dict = dict()
    file_list = dataset.get_file_list()
    for name, (_, g_t) in enumerate(dataset):
        enroll_part = len(g_t._tracks) // 2
        segments = list(g_t._tracks.keys()[:enroll_part])
        step = 10
        window = Segment(segments[0].start, segments[0].start + 5)
        intersect = []
        for seg in segments:
            if seg.intersects(window):
                speaker = list(g_t._tracks[seg].values())[0]
                intersect.append((speaker, file_list[name], seg))
        while window.end < segments[-1].start:
            if len(intersect) == 1:
                speaker = intersect[0][0]
                if speaker not in enroll_dict:
                    enroll_dict[speaker] = list()
                name_f = intersect[0][1]
                seg = intersect[0][2]
                new_start = max(seg.start, window.start)
                new_end = min(seg.end, window.end)
                new_seg = Segment(new_start, new_end)
                enroll_dict[speaker].append((name_f, new_seg))
            window = Segment(window.start + step, window.end + step)
            intersect = []
            for seg in segments:
                if seg.intersects(window):
                    speaker = list(g_t._tracks[seg].values())[0]
                    intersect.append((speaker, file_list[name], seg))
    final_enroll = dict()
    for speaker in enroll_dict:
        final_enroll[speaker] = list()
        duration = 0
        lst_enroll = list()
        idx_enroll = 0
        for file_name, enroll in enroll_dict[speaker]:
            start = enroll.start
            if duration + enroll.duration > 15:
                end = 15 + start - duration
                lst_enroll.append((file_name, Segment(start, end)))
                final_enroll[speaker].append((idx_enroll, lst_enroll))
                lst_enroll = list()
                duration = 0
                idx_enroll += 1
            else:
                end = enroll.end
                lst_enroll.append((file_name, Segment(start,end)))
                duration += enroll.duration
    return final_enroll


def get_test(dataset):
    test_dict = dict()
    file_list = dataset.get_file_list()
    for name, (_, g_t) in enumerate(dataset):
        test_part = len(g_t._tracks) // 2
        segments = list(g_t._tracks.keys()[test_part:])
        num_test = 0
        dict_enroll = dict()
        step = 40
        window = Segment(segments[0].start, segments[0].start + 60)
        intersect = []
        for seg in segments:
            if seg.intersects(window):
                intersect.append(seg)
                speaker = list(g_t._tracks[seg].values())[0]
                if speaker not in dict_enroll:
                    dict_enroll[speaker] = list()
                new_start = max(seg.start, window.start)
                new_end = min(seg.end, window.end)
                new_seg = Segment(new_start, new_end)
                dict_enroll[speaker].append(new_seg)
        while intersect:
            min_dur = float('inf')
            max_dur = 0
            min_start = float('inf')
            max_end = 0
            for speak in dict_enroll:
                dur = 0
                for s in dict_enroll[speak]:
                    dur += s.duration
                    min_start = min(min_start, s.start)
                    max_end = max(max_end, s.end)
                min_dur = min(dur, min_dur)
                max_dur = max(dur, max_dur)
            if max_dur / min_dur < 5 and max_end - min_start > 40:
                if file_list[name] not in test_dict:
                    test_dict[file_list[name]] = dict()
                test_dict[file_list[name]][num_test] = dict_enroll
                num_test += 1
                if num_test == 5:
                    break
            dict_enroll = dict()
            window = Segment(window.start + step, window.end + step)
            intersect = []
            for seg in segments:
                if seg.intersects(window):
                    intersect.append(seg)
                    speaker = list(g_t._tracks[seg].values())[0]
                    if speaker not in dict_enroll:
                        dict_enroll[speaker] = list()
                    new_start = max(seg.start, window.start)
                    new_end = min(seg.end, window.end)
                    new_seg = Segment(new_start, new_end)
                    dict_enroll[speaker].append(new_seg)
    return test_dict


def get_pairs(enrolls, test_dict):
    pairs = list()
    num = 0
    for name_file in test_dict:
        for num_test in test_dict[name_file]:
            speakers = list(test_dict[name_file][num_test].keys())
            for each in speakers:
                length = len(enrolls[each])
                enroll = enrolls[each][num % length][1]
                pairs.append((each, enroll, name_file, test_dict[name_file][num_test]))
    return pairs


def make_files(dataset, pairs):
    file_list = dataset.get_file_list()
    root = os.getcwd()
    dir_name = 'AMI_benchmark'
    path_bench = os.path.join(root, dir_name)
    if not os.path.exists(path_bench):
        os.mkdir(path_bench)
    for i, (speaker, enroll, name_file, tests) in enumerate(pairs):
        name_pair = 'test' + str(i)
        path_test = os.path.join(path_bench, name_pair)
        if not os.path.exists(path_test):
            os.mkdir(path_test)
        min_start = float('inf')
        max_end = 0
        for sp in tests:
            for seg in tests[sp]:
                min_start = min(min_start, seg.start)
                max_end = max(max_end, seg.end)
        audio, _ = dataset[file_list.index(name_file)]
        y = audio[int(SAMPLE_RATE * min_start): int(SAMPLE_RATE * max_end)]
        path_audio = os.path.join(path_test, 'audio.wav')
        sf.write(path_audio, y, SAMPLE_RATE)
        path_rttm = os.path.join(path_test, 'annot.rttm')
        with open(path_rttm, 'wb') as f:
            for sp in tests:
                for seg in tests[sp]: 
                    beg = seg.start - min_start
                    fields = ['SPEAKER', name_pair, '1', str(float(beg)), str(float(seg.duration)), 
                              '<NA>', '<NA>', sp, '<NA>', '<NA>']
                    line = ' '.join(fields)
                    f.write(line.encode('utf-8'))
                    f.write(b'\n')
        path_enroll = os.path.join(path_test, speaker + '.wav')
        y = None
        for en in enroll:
            audio, _ = dataset[file_list.index(en[0])]
            audio = audio[int(SAMPLE_RATE * en[1].start): int(SAMPLE_RATE * en[1].end)]
            y = np.concatenate((y, audio)) if y is not None else audio
        sf.write(path_enroll, y, SAMPLE_RATE)


DATA_ROOT = sys.argv[1]                  # path to audio
GROUND_TRUTH = sys.argv[2]               # path to annotations
SAMPLE_RATE = 16_000

test_data = AMITest(DATA_ROOT, GROUND_TRUTH)
enrolls = get_enroll(test_data)
test_dict = get_test(test_data)
pairs = get_pairs(enrolls, test_dict)
make_files(test_data, pairs)
