export CUDA_VISIBLE_DEVICES="0"

python \
    decode.py \
    --epoch 100 \
    --use-averaged-model true \
    --avg 20 \
    --causal 1 \
    --chunk-size 32 \
    --left-context-frames 256 \
    --exp-dir ./exp_feature_adamw \
    --max-duration 100 \
    --lang-type bpe \
    --lang /home/m-wu/proj/ASR/dataprocessing/bpe \
    --bpe-model /home/m-wu/proj/ASR/dataprocessing/bpe/bpe.model \
    --manifest-dir dataprocessing/cuts \
    --max-sym-per-frame 10 \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --use-pretrained false \
    --use-raw-wav false \
    --encoder-embed-type conv \
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