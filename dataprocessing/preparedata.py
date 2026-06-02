from pathlib import Path
from lhotse import RecordingSet
from lhotse import SupervisionSet
from lhotse import Recording
from lhotse import SupervisionSegment

import re
from pathlib import Path

from sklearn.model_selection import train_test_split

def load_trs_for_asr(trs_file):
    data = []

    pattern = re.compile(
        r'(\S+)\s+(\S+)\s+"(.*?)"\s+-from\s+([\d.]+)\s+-to\s+([\d.]+)'
    )

    with open(trs_file, encoding="utf-8") as f:
        idx = 0

        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            m = pattern.match(line)
            if m is None:
                continue

            _, wav_path, text, start, end = m.groups()

            data.append({
                "id": idx,
                "audio_path": wav_path,
                "start": float(start),
                "end": float(end),
                "text": text,
            })
            
            idx += 1

    return data

def save_cuts(wav_files, starts, ends, texts, output_dir):
    recordings = []
    supervisions = []

    for wav, start, end, text in zip(wav_files, starts, ends, texts):
        wav = Path("/ctd/SpeechData/Trainset/Japanese") / wav
        recording = Recording.from_file(str(wav))

        supervision = SupervisionSegment(
            id=f"{wav.stem}-{start:.2f}-{end:.2f}",
            recording_id=recording.id,
            start=start,
            duration=(end - start),
            text=text,
            language="Japanese",
        )

        recordings.append(recording)
        supervisions.append(supervision)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    train_rec, test_rec = train_test_split(recordings, test_size=0.4)
    test_rec, dev_rec = train_test_split(test_rec, test_size=0.5)

    train_sup, test_sup = train_test_split(supervisions, test_size=0.4)
    test_sup, dev_sup = train_test_split(test_sup, test_size=0.5)

    RecordingSet.from_recordings(train_rec).to_file(output_dir / "recordings_train.jsonl.gz")
    SupervisionSet.from_segments(train_sup).to_file(output_dir / "supervisions_train.jsonl.gz")

    RecordingSet.from_recordings(test_rec).to_file(output_dir / "recordings_test.jsonl.gz")
    SupervisionSet.from_segments(test_sup).to_file(output_dir / "supervisions_test.jsonl.gz")

    RecordingSet.from_recordings(dev_rec).to_file(output_dir / "recordings_dev.jsonl.gz")
    SupervisionSet.from_segments(dev_sup).to_file(output_dir / "supervisions_dev.jsonl.gz")

if __name__ == "__main__":
    trs_dir = "/ctd/SpeechData/Trainset/Japanese/E2E/ACP/16k/trs/20250515/original"
    trs_file = "/ctd/SpeechData/Trainset/Japanese/E2E/ACP/16k/trs/20250515/original/ACP_20250515_01.trs"
    output_dir = "/home/m-wu/proj/ASR/dataprocessing/cuts"
    data = load_trs_for_asr(trs_file)
    wav_files = [d["audio_path"] for d in data]
    texts = [d["text"] for d in data]
    starts = [d["start"] for d in data]
    ends = [d["end"] for d in data]
    save_cuts(wav_files, starts, ends, texts, output_dir)
