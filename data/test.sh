python data/test_dual_channel_datamodule.py \
  --manifest-dir-ch0 /home/m-wu/proj/ASR/dataprocessing/cuts \
  --manifest-dir-ch1 /home/m-wu/proj/ASR/dataprocessing/cuts \
  --split train \
  --num-batches 2 \
  --max-duration 500 \
  --num-workers 0 \
  --return-cuts true