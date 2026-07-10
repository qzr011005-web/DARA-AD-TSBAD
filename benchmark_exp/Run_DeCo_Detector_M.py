# -*- coding: utf-8 -*-
"""Run modified DeCo on TSB-AD-M with the official TSB-AD score-evaluation protocol."""

import argparse
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.models.DeCo import DeCo
from TSB_AD.utils.slidingWindows import find_length_rank


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def seed_everything(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Run modified DeCo on TSB-AD-M")
    parser.add_argument("--dataset_dir", type=str, default="./Datasets/TSB-AD-M/")
    parser.add_argument("--file_lsit", type=str, default="./Datasets/File_List/TSB-AD-M-Eva.csv")
    parser.add_argument("--score_dir", type=str, default="./eval/score/multi/")
    parser.add_argument("--save_dir", type=str, default="./eval/metrics/multi/")
    parser.add_argument("--save", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--win_size", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fc_dropout", type=float, default=0.1)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--multi_kernel_sizes", type=str, default="9,17,25")
    return parser.parse_args()


def run_one_file(args, filename: str):
    file_path = os.path.join(args.dataset_dir, filename)
    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df["Label"].astype(int).to_numpy()
    train_index = int(filename.split(".")[0].split("_")[-3])
    data_train = data[:train_index, :]
    sliding_window = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

    detector = DeCo(
        win_size=args.win_size,
        pred_len=args.pred_len,
        input_c=data.shape[1],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        validation_size=args.validation_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        fc_dropout=args.fc_dropout,
        patch_len=args.patch_len,
        stride=args.stride,
        multi_kernel_sizes=args.multi_kernel_sizes,
    )
    detector.fit(data_train)
    score = detector.decision_function(data)
    if len(score) != len(label):
        raise RuntimeError(f"score length {len(score)} != label length {len(label)} for {filename}")
    return score, label, sliding_window


def main():
    args = parse_args()
    seed_everything(args.seed)
    print("CUDA available: ", torch.cuda.is_available())
    print("cuDNN version: ", torch.backends.cudnn.version())

    ad_name = "DeCo"
    target_dir = os.path.join(args.score_dir, ad_name)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(target_dir, f"000_run_{ad_name}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    file_list = pd.read_csv(args.file_lsit)["file_name"].values
    rows = []
    metric_keys = None
    for filename in file_list:
        score_file = os.path.join(target_dir, filename.split(".")[0] + ".npy")
        if os.path.exists(score_file):
            continue
        print(f"Processing:{filename} by {ad_name}")
        start_time = time.time()
        try:
            output, label, sliding_window = run_one_file(args, filename)
            run_time = time.time() - start_time
            np.save(score_file, output)
            logging.info(f"Success at {filename} using {ad_name} | Time cost: {run_time:.3f}s at length {len(label)}")

            if args.save:
                # TSB-AD protocol: evaluate raw anomaly score directly. No q/d, no bidSPOT, no dynamic threshold.
                evaluation_result = get_metrics(output, label, slidingWindow=sliding_window)
                print("evaluation_result: ", evaluation_result)
                metric_keys = list(evaluation_result.keys())
                rows.append([filename, run_time] + list(evaluation_result.values()))
        except Exception as exc:
            run_time = time.time() - start_time
            logging.exception(f"Failed at {filename}: {exc}")
            if args.save:
                if metric_keys is None:
                    metric_keys = [
                        "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC", "Standard-F1",
                        "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F",
                    ]
                rows.append([filename, run_time] + [0] * len(metric_keys))

        if args.save and rows:
            columns = ["file", "Time"] + metric_keys
            pd.DataFrame(rows, columns=columns).to_csv(os.path.join(args.save_dir, f"{ad_name}.csv"), index=False)


if __name__ == "__main__":
    main()
