import os
import librosa
from pyannote.database.util import load_rttm
from pyannote.core import Segment
import soundfile as sf

SAMPLE_RATE = 16000
def make_enrolls(path_corpus, path_rttm):
    enrolls = dict()
    for _, _, filenames in os.walk(path_rttm):
        for filename in filenames:
            # Load files
            rttm_name = os.path.join(path_rttm, filename)
            _, groundtruth = load_rttm(rttm_name).popitem()
            enroll_part = len(groundtruth._tracks) // 6
            segments = list(groundtruth._tracks.keys()[:enroll_part])
            upper_time = 10 * 60

            # Init window
            start = 0
            win_len = 5
            window = Segment(start, start + win_len)
            intersect = []
            for seg in segments:
                if seg.intersects(window):
                    speaker = list(groundtruth._tracks[seg].values())[0]
                    intersect.append((speaker, filename[:-5], seg))
            # Iterate over groundtruth
            while window.end < min(upper_time, segments[-1].start):
                if len(intersect) == 1:
                    speaker = intersect[0][0]
                    if speaker not in enrolls:
                        enrolls[speaker] = list()
                    name_f = intersect[0][1]
                    seg = intersect[0][2]
                    new_start = max(seg.start, window.start)
                    new_end = min(seg.end, window.end)
                    new_seg = Segment(new_start, new_end)
                    enrolls[speaker].append((name_f, new_seg))
                window = Segment(window.start + win_len, window.end + win_len)
                intersect = []
                for seg in segments:
                    if seg.intersects(window):
                        speaker = list(groundtruth._tracks[seg].values())[0]
                        intersect.append((speaker, filename[:-5], seg))
    return enrolls


def make_trainpart(root, path_corpus, path_rttm, enrolls):
    datadir = os.path.join(root, 'train_vad')
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    for _, dirnames, _ in os.walk(path_corpus):
        for dirname in dirnames:
            if dirname == 'audio':
                continue
            # Create folder with new train data
            dirsession = os.path.join(datadir, dirname)
            if not os.path.exists(dirsession):
                os.mkdir(dirsession)
            
            # Load ami train part
            name_file = os.path.join(path_corpus, dirname, 'audio', 
                                     dirname + '.Mix-Headset.wav')
            audio, _ = librosa.load(name_file, sr=SAMPLE_RATE)

            # Cut audio
            audio = audio[SAMPLE_RATE*10*60:]
            
            # Save audio in new dir
            path_audio = os.path.join(dirsession, 'audio.wav')
            sf.write(path_audio, audio, SAMPLE_RATE)
    
    for _, _, filenames in os.walk(path_rttm):
        for filename in filenames:
            # Load files
            rttm_name = os.path.join(path_rttm, filename)
            _, groundtruth = load_rttm(rttm_name).popitem()
            segments = list(groundtruth._tracks.keys())

            # Delete all segments with seg.end < 10 min
            for seg in segments:
                if seg.end < 10 * 60:
                    del groundtruth._tracks[seg]
                elif seg.start < 10 * 60:
                    new_seg = Segment(10 * 60, seg.end)
                    info = groundtruth._tracks[seg]
                    del groundtruth._tracks[seg]
                    groundtruth._tracks[new_seg] = info
            
            # Create new rttm files
            path_annot = os.path.join(datadir, filename[:-5], filename)
            with open(path_annot, 'wb') as f:
                for idx, seg in enumerate(groundtruth._tracks):
                    beg = seg.start - 10*60
                    speaker = list(groundtruth._tracks[seg].values())[0]
                    fields = ['SPEAKER', filename[:-5], '1', str(float(beg)), str(float(seg.duration)), 
                              '<NA>', '<NA>', speaker, '<NA>', '<NA>']
                    line = ' '.join(fields)
                    f.write(line.encode('utf-8'))
                    f.write(b'\n')
            
            # Add enrolls to datadir
            speakers = set()
            for seg in groundtruth._tracks:
                speaker = list(groundtruth._tracks[seg].values())[0]
                speakers.add(speaker)
            for speaker in speakers:
                if speaker not in enrolls:
                    continue
                for info in enrolls[speaker]:
                    if info[0] == filename[:-5]:
                        continue
                    elif info[1].duration > 4:
                        name_file = os.path.join(path_corpus, info[0], 'audio', 
                                     info[0] + '.Mix-Headset.wav')
                        audio, _ = librosa.load(name_file, sr=SAMPLE_RATE)
                        audio = audio[int(SAMPLE_RATE * info[1].start): int(SAMPLE_RATE * info[1].end)]
                        name_file = os.path.join(datadir, filename[:-5],
                                                 'enroll_' + speaker + '.wav')
                        sf.write(name_file, audio, SAMPLE_RATE)
                        break
    
    