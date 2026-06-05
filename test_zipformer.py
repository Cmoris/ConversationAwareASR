import torch
from model.zipformer import Zipformer2

x = torch.randn((2,16000))
model = Zipformer2.from_hf_pretrained()
print(model)
print(model())