#!/usr/bin/env python3
"""
Smoke test for DualAsrFbankDataModule / DualChannelK2SpeechRecognitionDataset.

Example:
  python test_dual_channel_datamodule.py \
    --manifest-dir-ch0 data/manifests_A \
    --manifest-dir-ch1 data/manifests_B \
    --split train \
    --num-batches 2 \
    --max-duration 30 \
    --num-workers 0 \
    --return-cuts true

Assumption:
  This script is placed next to dual_channel_datamodule.py, or that module is
  importable from PYTHONPATH.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from lhotse import CutSet, load_manifest_lazy

from dual_channel_datamodule import DualAsrFbankDataModule


SPLIT_TO_FILE = {
    "train": "reazonspeech_cuts_train.jsonl.gz",
    "dev": "reazonspeech_cuts_dev.jsonl.gz",
    "valid": "reazonspeech_cuts_dev.jsonl.gz",
    "test": "reazonspeech_cuts_test.jsonl.gz",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got: {v}")


def iter_first(cuts: CutSet, n: int):
    for i, cut in enumerate(cuts):
        if i >= n:
            break
        yield cut


def check_manifest_files(args) -> Tuple[Path, Path]:
    filename = SPLIT_TO_FILE[args.split]
    ch0_path = args.manifest_dir_ch0 / filename
    ch1_path = args.manifest_dir_ch1 / filename
    if not ch0_path.is_file():
        raise FileNotFoundError(f"Channel 0 manifest not found: {ch0_path}")
    if not ch1_path.is_file():
        raise FileNotFoundError(f"Channel 1 manifest not found: {ch1_path}")
    print(f"[OK] ch0 manifest: {ch0_path}")
    print(f"[OK] ch1 manifest: {ch1_path}")
    return ch0_path, ch1_path


def check_pairing(cuts0: CutSet, cuts1: CutSet, num_check: int, duration_tol: float):
    """
    Check that the first `num_check` cuts in ch0 can be found in ch1 by id.
    For large lazy manifests, this avoids materializing all cuts unless needed.
    """
    print(f"\n[Pairing check] building ch1 id index from lazy manifest ...")
    cuts1_by_id: Dict[str, object] = {c.id: c for c in cuts1}
    print(f"[Pairing check] ch1 indexed cuts: {len(cuts1_by_id)}")

    missing: List[str] = []
    duration_mismatch: List[Tuple[str, float, float]] = []
    checked = 0

    for c0 in iter_first(cuts0, num_check):
        checked += 1
        c1 = cuts1_by_id.get(c0.id)
        if c1 is None:
            missing.append(c0.id)
            continue
        if abs(c0.duration - c1.duration) > duration_tol:
            duration_mismatch.append((c0.id, c0.duration, c1.duration))

    print(f"[Pairing check] checked first {checked} ch0 cuts")
    if missing:
        print(f"[BAD] missing in ch1: {len(missing)} examples, first 10 = {missing[:10]}")
    else:
        print("[OK] all checked ch0 cut ids exist in ch1")

    if duration_mismatch:
        print(
            f"[WARN] duration mismatch > {duration_tol}s: "
            f"{len(duration_mismatch)} examples, first 5 = {duration_mismatch[:5]}"
        )
    else:
        print(f"[OK] all checked paired durations are within {duration_tol}s")

    if missing:
        raise RuntimeError("Pairing check failed: some ch0 cut ids are missing in ch1.")


def tensor_info(name: str, x):
    if torch.is_tensor(x):
        print(f"  {name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
    else:
        print(f"  {name}: type={type(x)}")


def print_supervision_info(name: str, sups, max_items: int = 3):
    print(f"  {name}: keys={list(sups.keys())}")
    for key in ("text", "cut", "sequence_idx", "start_frame", "num_frames"):
        if key not in sups:
            continue
        value = sups[key]
        if key == "cut":
            try:
                ids = [c.id for c in value[:max_items]]
            except Exception:
                ids = str(type(value))
            print(f"    {key}: {ids}")
        elif torch.is_tensor(value):
            print(f"    {key}: shape={tuple(value.shape)}, first={value[:max_items].tolist()}")
        else:
            print(f"    {key}: {value[:max_items] if hasattr(value, '__getitem__') else value}")


def check_dataloader(args):
    dm = DualAsrFbankDataModule(args)

    if args.split == "train":
        cuts_pair = dm.train_cuts()
        dl = dm.train_dataloaders(cuts_pair)
    elif args.split in ("dev", "valid"):
        cuts_pair = dm.valid_cuts()
        dl = dm.valid_dataloaders(cuts_pair)
    else:
        cuts_pair = dm.test_cuts()
        dl = dm.test_dataloaders(cuts_pair)

    print(f"\n[Dataloader check] split={args.split}")
    for batch_idx, batch in enumerate(dl):
        if batch_idx >= args.num_batches:
            break

        print(f"\nBatch {batch_idx}")
        print(f"  batch keys: {list(batch.keys())}")
        tensor_info("inputs", batch["inputs"])
        tensor_info("input_lens", batch["input_lens"])
        tensor_info("inputs_ch0", batch["inputs_ch0"])
        tensor_info("inputs_ch1", batch["inputs_ch1"])
        tensor_info("input_lens_ch0", batch["input_lens_ch0"])
        tensor_info("input_lens_ch1", batch["input_lens_ch1"])

        x = batch["inputs"]
        assert x.ndim in (3, 4), f"Expected [B,2,T] or [B,2,T,F], got {tuple(x.shape)}"
        assert x.shape[1] == 2, f"Expected channel dim == 2, got {tuple(x.shape)}"
        assert torch.equal(batch["inputs_ch0"], x[:, 0]), "inputs_ch0 != inputs[:, 0]"
        assert torch.equal(batch["inputs_ch1"], x[:, 1]), "inputs_ch1 != inputs[:, 1]"
        assert torch.equal(
            batch["input_lens"],
            torch.maximum(batch["input_lens_ch0"], batch["input_lens_ch1"]),
        ), "input_lens is not max(ch0, ch1)"

        print_supervision_info("supervisions_ch0", batch["supervisions_ch0"])
        print_supervision_info("supervisions_ch1", batch["supervisions_ch1"])

    print("\n[OK] dataloader smoke test finished.")


def get_parser():
    parser = argparse.ArgumentParser()

    # Reuse the datamodule's own options so defaults stay consistent.
    DualAsrFbankDataModule.add_arguments(parser)

    parser.add_argument(
        "--split",
        choices=["train", "dev", "valid", "test"],
        default="train",
        help="Which split to test.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=2,
        help="Number of batches to draw from the dataloader.",
    )
    parser.add_argument(
        "--num-pair-check",
        type=int,
        default=100,
        help="Number of ch0 cuts to check for id/duration pairing before loading batches.",
    )
    parser.add_argument(
        "--duration-tol",
        type=float,
        default=0.05,
        help="Warn when paired cuts have duration difference larger than this many seconds.",
    )
    parser.add_argument(
        "--skip-pair-check",
        type=str2bool,
        default=False,
        help="Skip manifest id/duration pairing check.",
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args()

    if args.manifest_dir_ch0 is None:
        args.manifest_dir_ch0 = args.manifest_dir
    if args.manifest_dir_ch1 is None:
        raise ValueError("Please set --manifest-dir-ch1")

    # Make test deterministic and less surprising.
    args.shuffle = False
    args.drop_last = False
    args.enable_spec_aug = False
    args.enable_musan = False
    args.concatenate_cuts = False

    check_manifest_files(args)

    if not args.skip_pair_check:
        filename = SPLIT_TO_FILE[args.split]
        cuts0 = load_manifest_lazy(args.manifest_dir_ch0 / filename)
        cuts1 = load_manifest_lazy(args.manifest_dir_ch1 / filename)
        check_pairing(cuts0, cuts1, args.num_pair_check, args.duration_tol)

    check_dataloader(args)


if __name__ == "__main__":
    main()
