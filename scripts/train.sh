export CUDA_VISIBLE_DEVICES="0,1,2,3"

python ./train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --use-pretrained false \
  --exp-dir ./exp \
  --causal 1 \
  --max-duration 250 \
  --enable-spec-aug false \
  --lang dataprocessing/bpe \
  --manifest-dir dataprocessing/cuts \
  --bpe-model dataprocessing/bpe/bpe.model \
  --num-encoder-layers 2,2,3,4,3,2 \
  --downsampling-factor 1,2,4,8,4,2 \
  --feedforward-dim 512,768,1024,1536,1024,768 \
  --num-heads 4,4,4,8,4,4 \
  --encoder-dim 192,256,448,768,448,192 \
  --query-head-dim 32 \
  --value-head-dim 12 \
  --pos-head-dim 4 \
  --pos-dim 48 \
  --encoder-unmasked-dim 192,192,256,256,256,192 \
  --cnn-module-kernel 31,31,15,15,15,31 \