"""Run the GraphCBM experiment on the bundled MUTAG dataset."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 0
NOISE_RATE = 0.30
MIXUP_GRAPHS = 200


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_fold():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    random.seed(SEED)


def summarize_histories(histories):
    epoch_accuracies = [
        [Fraction(row["test_correct"], row["test_size"]) for row in rows]
        for rows in zip(*histories, strict=True)
    ]
    best_epoch = max(range(len(epoch_accuracies)), key=lambda index: sum(epoch_accuracies[index]))
    best_values = epoch_accuracies[best_epoch]
    return {
        "epoch": best_epoch,
        "mean_acc": float(statistics.mean(best_values)),
        "population_std": statistics.pstdev([float(value) for value in best_values]),
        "folds": len(histories),
    }


def run(args):
    from augmentation import append_mixup_graphs
    from data import add_training_noise, load_mutag, split_folds
    from trainer import Trainer, WARMUP_EPOCHS

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    save_json(output / "configuration.json", {
        "dataset": "MUTAG", "seed": SEED, "uniform_noise": NOISE_RATE,
        "folds": list(range(10)) if args.fold is None else [args.fold],
        "pretrain_epochs": args.pretrain_epochs, "epochs": args.epochs, "warmup_epochs": WARMUP_EPOCHS,
        "mixup_graphs": MIXUP_GRAPHS, "pretrain_views": ["identity", "node_drop"],
        "device": str(device), "threads": args.threads, "torch": torch.__version__,
    })

    graphs, node_features, num_classes, text = load_mutag(ROOT / "dataset")
    histories = []
    for fold, (clean_train, test) in enumerate(split_folds(graphs, SEED)):
        if args.fold is not None and fold != args.fold:
            continue
        seed_fold()
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir()
        train = add_training_noise(clean_train, num_classes, NOISE_RATE, SEED)
        print(f"MUTAG fold={fold} train={len(train)} test={len(test)}", flush=True)
        save_json(fold_dir / "split.json", {
            "train_graph_ids": [graph.graph_id for graph in train],
            "test_graph_ids": [graph.graph_id for graph in test],
            "observed_labels": [graph.label for graph in train],
            "flipped_count": sum(a.label != b.label for a, b in zip(clean_train, train)),
        })

        pretraining_graphs = append_mixup_graphs(train, MIXUP_GRAPHS)
        trainer = Trainer(node_features, num_classes, text, device)
        pretraining = trainer.pretrain(pretraining_graphs, args.pretrain_epochs, fold_dir / "pretrained_gin.pt")
        pretraining["num_graphs"] = len(pretraining_graphs)
        save_json(fold_dir / "pretraining.json", pretraining)
        with (fold_dir / "history.jsonl").open("w", encoding="utf-8") as record:
            def record_epoch(row):
                record.write(json.dumps(row, allow_nan=False) + "\n")
                record.flush()
                print(f"Epoch {row['epoch']:03d}  loss={row['train_loss']:.5f}  "
                      f"test_acc={row['test_acc']:.5f}  selected={row['selected']}", flush=True)

            trainer.finetune(train, test, args.epochs, record_epoch)

        torch.save({"student": trainer.student.state_dict(), "gin": trainer.pretraining.gin.state_dict()},
                   fold_dir / "final_model.pt")
        histories.append([json.loads(line) for line in (fold_dir / "history.jsonl").read_text().splitlines()])

    result = summarize_histories(histories)
    result["full_protocol"] = args.fold is None and args.epochs == 350 and args.pretrain_epochs == 1000
    save_json(output / "summary.json", result)
    label = "RESULT" if result["full_protocol"] else "Run summary"
    print(f"{label}  MUTAG  epoch={result['epoch']}  "
          f"ACC={100 * result['mean_acc']:.2f}% ± {100 * result['population_std']:.2f}%", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for checkpoints and fold histories")
    parser.add_argument("--fold", type=int, choices=range(10), help="Run a single fold. Default: all 10 folds")
    parser.add_argument("--pretrain-epochs", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    args = parser.parse_args()
    if min(args.pretrain_epochs, args.epochs, args.threads) < 1:
        parser.error("Epoch counts and thread count must be positive")
    if args.output.exists():
        parser.error("Choose a new output directory")
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
