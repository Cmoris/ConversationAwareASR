from pathlib import Path
from lhotse import RecordingSet
from lhotse import SupervisionSet
from lhotse import Recording
from lhotse import SupervisionSegment

import re
from pathlib import Path


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

def save_cuts(wav_files, texts, output_dir):
    recordings = []
    supervisions = []

    for wav, text in zip(wav_files, texts):
        wav = Path("/ctd/SpeechData/Trainset/Japanese") / wav
        recording = Recording.from_file(str(wav))

        supervision = SupervisionSegment(
            id=wav.stem,
            recording_id=wav.stem,
            start=0,
            duration=recording.duration,
            text=text,
            language="Japanese",
        )

        recordings.append(recording)
        supervisions.append(supervision)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    RecordingSet.from_recordings(recordings).to_file(output_dir / "recordings.jsonl.gz")
    SupervisionSet.from_segments(supervisions).to_file(output_dir / "supervisions.jsonl.gz")

if __name__ == "__main__":
    trs_dir = "/ctd/SpeechData/Trainset/Japanese/E2E/ACP/16k/trs/20250515/original"
    trs_file = "/ctd/SpeechData/Trainset/Japanese/E2E/ACP/16k/trs/20250515/original/ACP_20250515_01.trs"
    output_dir = "/home/m-wu/proj/ASR/dataprocessing/cuts"
    data = load_trs_for_asr(trs_file)
    wav_files = [d["audio_path"] for d in data]
    texts = [d["text"] for d in data]
    save_cuts(wav_files, texts, output_dir)
