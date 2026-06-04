from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy

manifest_path = "/home/m-wu/proj/ASR/dataprocessing/cuts/reazonspeech_cuts_dev.jsonl.gz"
cutset = load_manifest(manifest_path)
print(cutset[0].load_audio())