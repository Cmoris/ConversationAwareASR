# import torch
# import k2
# print(torch.__version__)
# print(k2.k2_torch_version)
# print(k2.k2_torch_cuda_version)

import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModel

feature_extractor = AutoFeatureExtractor.from_pretrained("reazon-research/japanese-zipformer-base-k2")
model = AutoModel.from_pretrained("reazon-research/japanese-zipformer-base-k2", trust_remote_code=True)

audio_file = "/ctd/SpeechData/Trainset/Japanese/E2E/ACP/16k/audio/20250515/01954c2853f60a3036af94c6_20250228_194426.wav"

audio, sr = librosa.load(audio_file, sr=16_000)
inputs = feature_extractor(
    audio,
    return_tensors="pt",
    sampling_rate=sr,
)
# inputs['padding_mask'] = (inputs['input_values'] != 0)

print(audio.shape)
print(inputs['input_values'].size())
print(inputs.keys())
print(model.config)
# print(model)


with torch.inference_mode():
    outputs = model(**inputs)
print(outputs)