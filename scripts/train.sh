export CUDA_VISIBLE_DEVICES="0"

python ./train.py \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir ./exp \
  --causal 1 \
  --max-duration 1000 \
  --lang dataprocessing/bpe \
  --manifest-dir dataprocessing/cuts \
  --bpe-model dataprocessing/bpe/bpe.model