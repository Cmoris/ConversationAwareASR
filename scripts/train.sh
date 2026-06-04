export CUDA_VISIBLE_DEVICES="0"

python ./train.py \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir ./exp \
  --causal 1 \
  --max-duration 1000 \
  --enable-spec-aug false \
  --lang dataprocessing/bpe \
  --manifest-dir dataprocessing/cuts \
  --bpe-model dataprocessing/bpe/bpe.model \
  --num-encoder-layers 
  --downsampling-factor
  --feedforward-dim
  --num-heads
  --encoder-dim "192,256,448,768,448,192"
  --query-head-dim
  --value-head-dim
  --pos-head-dim
  --pos-dim
  --encoder-unmasked-dim
  --cnn-module-kernel
  --decoder-dim
  --joiner-dim
  --chunk-size
  --left-context-frames