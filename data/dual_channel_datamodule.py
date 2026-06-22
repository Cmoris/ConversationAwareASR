import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from lhotse import CutSet, Fbank, FbankConfig, load_manifest_lazy
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import AudioSamples, OnTheFlyFeatures, PrecomputedFeatures
from torch.utils.data import DataLoader

from utils.utils import str2bool


class DualChannelK2SpeechRecognitionDataset:
    """
    Wrap two single-channel Lhotse CutSets into one dual-channel batch.

    The sampler samples from channel 0 CutSet. For each sampled cut id, this
    dataset fetches the cut with the same id from channel 1 CutSet, runs the
    normal K2SpeechRecognitionDataset pipeline on both sides, and returns:

        batch["inputs"]:        [B, 2, T] for AudioSamples
                              or [B, 2, T, F] for OnTheFlyFeatures
        batch["input_lens"]:    max(input_lens_ch0, input_lens_ch1)
        batch["inputs_ch0"] / batch["inputs_ch1"]
        batch["input_lens_ch0"] / batch["input_lens_ch1"]
        batch["supervisions_ch0"] / batch["supervisions_ch1"]

    `batch["supervisions"]` is kept as channel 0 for compatibility with
    existing icefall/k2 training code.
    """

    def __init__(
        self,
        cuts_ch1: CutSet,
        input_strategy,
        input_transforms=None,
        return_cuts: bool = False,
    ):
        self.cuts_ch1_by_id = {c.id: c for c in cuts_ch1}
        self.dataset_ch0 = K2SpeechRecognitionDataset(
            input_strategy=input_strategy,
            input_transforms=input_transforms or [],
            return_cuts=return_cuts,
        )
        self.dataset_ch1 = K2SpeechRecognitionDataset(
            input_strategy=input_strategy,
            input_transforms=input_transforms or [],
            return_cuts=return_cuts,
        )

    @staticmethod
    def _pad_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad tensor on its time dimension. Expected [B,T] or [B,T,F]."""
        cur_len = x.shape[1]
        if cur_len == target_len:
            return x
        pad_len = target_len - cur_len
        if pad_len < 0:
            return x[:, :target_len]
        if x.ndim == 2:      # [B, T]
            return F.pad(x, (0, pad_len))
        if x.ndim == 3:      # [B, T, F]
            return F.pad(x, (0, 0, 0, pad_len))
        raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

    def __getitem__(self, cuts_ch0: CutSet) -> Dict[str, Any]:
        missing = [c.id for c in cuts_ch0 if c.id not in self.cuts_ch1_by_id]
        if missing:
            raise KeyError(
                f"{len(missing)} cuts from channel 0 are missing in channel 1. "
                f"Examples: {missing[:5]}"
            )

        cuts_ch1 = CutSet.from_cuts(self.cuts_ch1_by_id[c.id] for c in cuts_ch0)

        batch_ch0 = self.dataset_ch0[cuts_ch0]
        batch_ch1 = self.dataset_ch1[cuts_ch1]

        x0 = batch_ch0["inputs"]
        x1 = batch_ch1["inputs"]
        target_len = max(x0.shape[1], x1.shape[1])
        x0 = self._pad_time(x0, target_len)
        x1 = self._pad_time(x1, target_len)

        batch = dict(batch_ch0)
        batch["inputs_ch0"] = x0
        batch["inputs_ch1"] = x1
        
        batch["input_lens_ch0"] = batch_ch0["supervisions"].get("num_frames", batch_ch0["supervisions"].get("num_samples", None))
        batch["input_lens_ch1"] = batch_ch1["supervisions"].get("num_frames", batch_ch1["supervisions"].get("num_samples", None))
        batch["supervisions_ch0"] = batch_ch0["supervisions"]
        batch["supervisions_ch1"] = batch_ch1["supervisions"]

        # [B, 2, T] for raw samples, [B, 2, T, F] for fbank/features.
        batch["inputs"] = torch.stack([x0, x1], dim=1)
        batch["input_lens"] = torch.maximum(
            batch["input_lens_ch0"], batch["input_lens_ch1"]
        )

        return batch


class DualAsrWavDataModule:
    """DataModule for dual-channel k2 ASR experiments."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="Options for preparing PyTorch DataLoaders from Lhotse CutSets.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/manifests"),
            help="Backward-compatible single manifest dir. Used as ch0 when --manifest-dir-ch0 is not set.",
        )
        group.add_argument(
            "--manifest-dir-ch0",
            type=Path,
            default=None,
            help="Path to channel 0/A manifest directory.",
        )
        group.add_argument(
            "--manifest-dir-ch1",
            type=Path,
            default=None,
            help="Path to channel 1/B manifest directory.",
        )
        group.add_argument("--max-duration", type=float, default=200.0)
        group.add_argument("--bucketing-sampler", type=str2bool, default=True)
        group.add_argument("--num-buckets", type=int, default=30)
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="Not recommended for dual-channel paired cuts; keep False unless you implement paired concatenation.",
        )
        group.add_argument("--duration-factor", type=float, default=1.0)
        group.add_argument("--gap", type=float, default=1.0)
        group.add_argument("--on-the-fly-feats", type=str2bool, default=False)
        group.add_argument("--shuffle", type=str2bool, default=True)
        group.add_argument("--drop-last", type=str2bool, default=True)
        group.add_argument("--return-cuts", type=str2bool, default=False)
        group.add_argument("--num-workers", type=int, default=2)
        group.add_argument("--enable-spec-aug", type=str2bool, default=True)
        group.add_argument("--spec-aug-time-warp-factor", type=int, default=80)
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=False,
            help="Disabled in this dual-channel version unless you implement paired CutMix.",
        )

    @property
    def manifest_dir_ch0(self) -> Path:
        return self.args.manifest_dir_ch0 or self.args.manifest_dir

    @property
    def manifest_dir_ch1(self) -> Path:
        if self.args.manifest_dir_ch1 is None:
            raise ValueError("Please set --manifest-dir-ch1 for dual-channel training.")
        return self.args.manifest_dir_ch1

    def _input_strategy(self):
        if self.args.on_the_fly_feats:
            return OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
        return AudioSamples()

    def _input_transforms(self, train: bool):
        input_transforms = []
        if train and self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        return input_transforms

    def _make_dataset(self, cuts_ch1: CutSet, train: bool):
        if self.args.concatenate_cuts:
            logging.warning(
                "--concatenate-cuts is ignored in the simple dual-channel dataset. "
                "Concatenation must be paired across both channels."
            )
        if self.args.enable_musan:
            logging.warning(
                "--enable-musan is ignored in the simple dual-channel dataset. "
                "CutMix must be applied pairwise to both channels."
            )
        return DualChannelK2SpeechRecognitionDataset(
            cuts_ch1=cuts_ch1,
            input_strategy=self._input_strategy(),
            input_transforms=self._input_transforms(train=train),
            return_cuts=self.args.return_cuts,
        )

    def train_dataloaders(
        self,
        cuts_train: Tuple[CutSet, CutSet],
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        cuts_musan: Optional[CutSet] = None,
    ) -> DataLoader:
        
        cuts_ch0, cuts_ch1 = cuts_train
        train = self._make_dataset(cuts_ch1, train=True)

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler on channel 0 cuts.")
            train_sampler = DynamicBucketingSampler(
                cuts_ch0,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler on channel 0 cuts.")
            train_sampler = SimpleCutSampler(
                cuts_ch0,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        return DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

    def valid_dataloaders(self, cuts_valid: Tuple[CutSet, CutSet]) -> DataLoader:
        cuts_ch0, cuts_ch1 = cuts_valid
        validate = self._make_dataset(cuts_ch1, train=False)
        valid_sampler = DynamicBucketingSampler(
            cuts_ch0,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        return DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

    def test_dataloaders(self, cuts: Tuple[CutSet, CutSet]) -> DataLoader:
        cuts_ch0, cuts_ch1 = cuts
        test = self._make_dataset(cuts_ch1, train=False)
        sampler = DynamicBucketingSampler(
            cuts_ch0,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        return DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )

    def _load_pair(self, split: str) -> Tuple[CutSet, CutSet]:
        filename = f"reazonspeech_cuts_{split}.jsonl.gz"
        logging.info(f"Loading dual-channel {split} cuts")
        cuts_ch0 = load_manifest_lazy(self.manifest_dir_ch0 / filename)
        cuts_ch1 = load_manifest_lazy(self.manifest_dir_ch1 / filename)
        return cuts_ch0, cuts_ch1

    @lru_cache()
    def train_cuts(self) -> Tuple[CutSet, CutSet]:
        return self._load_pair("train")

    @lru_cache()
    def valid_cuts(self) -> Tuple[CutSet, CutSet]:
        return self._load_pair("dev")

    @lru_cache()
    def test_cuts(self) -> Tuple[CutSet, CutSet]:
        return self._load_pair("test")


class DualAsrFbankDataModule:
    """DataModule for dual-channel k2 ASR experiments."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="Options for preparing PyTorch DataLoaders from Lhotse CutSets.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/manifests"),
            help="Backward-compatible single manifest dir. Used as ch0 when --manifest-dir-ch0 is not set.",
        )
        group.add_argument(
            "--manifest-dir-ch0",
            type=Path,
            default=None,
            help="Path to channel 0/A manifest directory.",
        )
        group.add_argument(
            "--manifest-dir-ch1",
            type=Path,
            default=None,
            help="Path to channel 1/B manifest directory.",
        )
        group.add_argument("--max-duration", type=float, default=200.0)
        group.add_argument("--bucketing-sampler", type=str2bool, default=True)
        group.add_argument("--num-buckets", type=int, default=30)
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="Not recommended for dual-channel paired cuts; keep False unless you implement paired concatenation.",
        )
        group.add_argument("--duration-factor", type=float, default=1.0)
        group.add_argument("--gap", type=float, default=1.0)
        group.add_argument("--on-the-fly-feats", type=str2bool, default=False)
        group.add_argument("--shuffle", type=str2bool, default=True)
        group.add_argument("--drop-last", type=str2bool, default=True)
        group.add_argument("--return-cuts", type=str2bool, default=False)
        group.add_argument("--num-workers", type=int, default=2)
        group.add_argument("--enable-spec-aug", type=str2bool, default=True)
        group.add_argument("--spec-aug-time-warp-factor", type=int, default=80)
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=False,
            help="Disabled in this dual-channel version unless you implement paired CutMix.",
        )

    @property
    def manifest_dir_ch0(self) -> Path:
        return self.args.manifest_dir_ch0 or self.args.manifest_dir

    @property
    def manifest_dir_ch1(self) -> Path:
        if self.args.manifest_dir_ch1 is None:
            raise ValueError("Please set --manifest-dir-ch1 for dual-channel training.")
        return self.args.manifest_dir_ch1

    def _input_strategy(self):
        if self.args.on_the_fly_feats:
            return OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
        return PrecomputedFeatures()

    def _input_transforms(self, train: bool):
        input_transforms = []
        if train and self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        return input_transforms

    def _make_dataset(self, cuts_ch1: CutSet, train: bool):
        if self.args.concatenate_cuts:
            logging.warning(
                "--concatenate-cuts is ignored in the simple dual-channel dataset. "
                "Concatenation must be paired across both channels."
            )
        if self.args.enable_musan:
            logging.warning(
                "--enable-musan is ignored in the simple dual-channel dataset. "
                "CutMix must be applied pairwise to both channels."
            )
        return DualChannelK2SpeechRecognitionDataset(
            cuts_ch1=cuts_ch1,
            input_strategy=self._input_strategy(),
            input_transforms=self._input_transforms(train=train),
            return_cuts=self.args.return_cuts,
        )

    def train_dataloaders(
        self,
        cuts_train: Tuple[CutSet, CutSet],
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        cuts_musan: Optional[CutSet] = None,
    ) -> DataLoader:
        
        cuts_ch0, cuts_ch1 = cuts_train
        train = self._make_dataset(cuts_ch1, train=True)

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler on channel 0 cuts.")
            train_sampler = DynamicBucketingSampler(
                cuts_ch0,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler on channel 0 cuts.")
            train_sampler = SimpleCutSampler(
                cuts_ch0,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        return DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

    def valid_dataloaders(self, cuts_valid: Tuple[CutSet, CutSet]) -> DataLoader:
        cuts_ch0, cuts_ch1 = cuts_valid
        validate = self._make_dataset(cuts_ch1, train=False)
        valid_sampler = DynamicBucketingSampler(
            cuts_ch0,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        return DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

    def test_dataloaders(self, cuts: Tuple[CutSet, CutSet]) -> DataLoader:
        cuts_ch0, cuts_ch1 = cuts
        test = self._make_dataset(cuts_ch1, train=False)
        sampler = DynamicBucketingSampler(
            cuts_ch0,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        return DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )

    def _load_pair(self, split: str) -> Tuple[CutSet, CutSet]:
        filename = f"reazonspeech_cuts_{split}.jsonl.gz"
        logging.info(f"Loading dual-channel {split} cuts")
        cuts_ch0 = load_manifest_lazy(self.manifest_dir_ch0 / filename)
        cuts_ch1 = load_manifest_lazy(self.manifest_dir_ch1 / filename)
        return cuts_ch0, cuts_ch1

    @lru_cache()
    def train_cuts(self) -> Tuple[CutSet, CutSet]:
        return self._load_pair("train")

    @lru_cache()
    def valid_cuts(self) -> Tuple[CutSet, CutSet]:
        return self._load_pair("dev")

    @lru_cache()
    def test_cuts(self) -> Tuple[CutSet, CutSet]:
        return self._load_pair("test")