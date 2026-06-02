import re
import numpy as np
from pathlib import Path

from torchcodec.decoders import AudioDecoder

SR = 16000

def read_audio(ele: dict):
    audio_decoder = AudioDecoder(source=ele['audio'], sample_rate=SR)
    audio_sr = audio_decoder.metadata.sample_rate
    audio_duration = audio_decoder.metadata.duration_seconds_from_header
    total_frames = int(audio_duration*audio_sr)
    audio_pts = np.linspace(1/audio_sr, audio_duration, total_frames)
    audio_start = ele.get("audio_start", None)
    audio_end = ele.get("audio_end", None)
    clip_idxs = None
    if audio_start is not None or audio_end is not None:
        audio_start = audio_pts[0] if not audio_start else audio_start
        audio_end = audio_pts[-1] if not audio_end else audio_end
        clip_idxs = ((audio_start <= audio_pts) & (audio_pts <= audio_end)).nonzero()[0]
        clip_pts = audio_pts[clip_idxs]
        total_frames = len(clip_pts)
    else:
        audio_start = 0
        audio_end = audio_duration
        
    nframes = int(total_frames/audio_sr*SR)
    nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]
    clip_pts = audio_pts[clip_idxs]
    clip = audio_decoder.get_samples_played_in_range(start_seconds=audio_start, stop_seconds=audio_end+1/SR).data
    
    return clip.squeeze(0), clip_pts, audio_sr

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
            breakpoint()
            idx += 1

    return data

if __name__ == "__main__":
    dir = "/ctd/SpeechData/Trainset/Japanese/E2E/ACP/16k/trs/20250515/original"
    trs_files = [f for f in Path(dir).glob("*.trs")]
    breakpoint()
    data = load_trs_for_asr(trs_files[0])