from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from pathlib import Path

manifest_dir = "/home/m-wu/proj/ASR/dataprocessing/cuts"
manifest_paths = [x for x in Path(manifest_dir).glob("*_cuts_*.jsonl.gz")]
for manifest_path in manifest_paths:
    cutset = load_manifest(manifest_path)
    print(len(cutset))