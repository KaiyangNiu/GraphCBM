"""Execute one fold of the GraphCBM pre-training and fine-tuning protocol."""

from copy import deepcopy
import random

import torch
import torch.nn.functional as F

from augmentation import node_drop_view
from correction import (
    classification_loss, confirm_corrections, initialize_soft_labels,
    update_soft_label_history, update_teacher,
)
from model import MultimodalClassifier, PretrainingModel, contrastive_loss
from prototypes import build_prototype_bank, score_prototypes

BATCH_SIZE = 64
WARMUP_EPOCHS = 30
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
PRETRAIN_PATIENCE = 5


def batches(graphs):
    parts = [graphs[i:i + BATCH_SIZE] for i in range(0, len(graphs), BATCH_SIZE)]
    if len(parts) > 1 and len(parts[-1]) == 1:
        parts[-2].extend(parts.pop())
    return parts


class Trainer:
    def __init__(self, input_size, num_classes, text_embeddings, device):
        self.device = device
        self.num_classes = num_classes
        self.text_embeddings = text_embeddings.to(device)
        self.pretraining = PretrainingModel(input_size, device).to(device)
        self.student = MultimodalClassifier(input_size, num_classes, device).to(device)

    def pretrain(self, graphs, epochs, checkpoint):
        optimizer = torch.optim.Adam(self.pretraining.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        augmented = node_drop_view(graphs)
        best_loss, best_epoch, stale = float("inf"), -1, 0
        self.pretraining.train()
        for epoch in range(epochs):
            loss = contrastive_loss(self.pretraining(graphs), self.pretraining(augmented))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite contrastive loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            value = loss.item()
            print(f"Pretrain {epoch:04d}  loss={value:.6f}", flush=True)
            if value < best_loss:
                best_loss, best_epoch, stale = value, epoch, 0
                torch.save(self.pretraining.state_dict(), checkpoint)
            else:
                stale += 1
            if stale > PRETRAIN_PATIENCE:
                break
        self.pretraining.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.pretraining.requires_grad_(False).eval()
        return {"best_epoch": best_epoch, "best_loss": best_loss, "epochs_run": epoch + 1}

    @torch.no_grad()
    def _prepare_corrections(self, groups, teacher, observed, history, streak, previous_class):
        self.student.eval()
        gin = self.pretraining.gin
        structure = torch.cat([self.student.structure(gin, group) for group in groups])
        semantic = self.text_embeddings[[graph.graph_id for group in groups for graph in group]]
        support = history.gather(1, observed[:, None]).squeeze(1)
        structure_bank = build_prototype_bank(structure, observed, support, self.num_classes)
        semantic_bank = build_prototype_bank(semantic, observed, support, self.num_classes)
        reference = torch.cat([teacher(gin, group, self.text_embeddings).softmax(dim=1) for group in groups])
        corrections = confirm_corrections(
            score_prototypes(structure, structure_bank), score_prototypes(semantic, semantic_bank),
            reference, observed, history, streak, previous_class,
        )
        counts = {"structure": [len(p.members) for p in structure_bank], "semantic": [len(p.members) for p in semantic_bank]}
        return corrections, counts

    @torch.no_grad()
    def _refresh_history(self, groups, teacher, history, max_step):
        gin = self.pretraining.gin
        largest_update = 0.0
        for group in groups:
            ids = [graph.sample_id for graph in group]
            original = teacher(gin, group, self.text_embeddings).softmax(dim=1)
            dropout = teacher(gin, group, self.text_embeddings, feature_dropout=0.10).softmax(dim=1)
            updated = update_soft_label_history(history[ids], original, dropout, max_step)
            largest_update = max(largest_update, (history[ids] - updated).abs().max().item())
            history[ids] = updated
        return largest_update

    @torch.no_grad()
    def _count_correct(self, graphs):
        self.student.eval()
        correct = 0
        for group in batches(graphs):
            predicted = self.student(self.pretraining.gin, group, self.text_embeddings).argmax(dim=1)
            labels = torch.tensor([graph.label for graph in group], device=self.device)
            correct += (predicted == labels).sum().item()
        return correct

    def finetune(self, train, test, epochs, record_epoch):
        gin = self.pretraining.gin
        optimizer = torch.optim.SGD([
            {"params": [self.student.node_prompt]},
            {"params": self.student.classifier.parameters(), "weight_decay": WEIGHT_DECAY},
            {"params": self.student.topology.parameters()},
            {"params": self.student.text_projector.parameters()},
        ], lr=LEARNING_RATE)
        teacher = deepcopy(self.student).requires_grad_(False).eval()
        observed = torch.tensor([graph.label for graph in train], device=self.device)
        history = initialize_soft_labels(observed, self.num_classes)
        streak, previous_class = torch.zeros_like(observed), torch.full_like(observed, -1)
        groups = batches(train)
        for epoch in range(epochs):
            corrections, counts = None, None
            if epoch >= WARMUP_EPOCHS:
                corrections, counts = self._prepare_corrections(groups, teacher, observed, history, streak, previous_class)
                streak, previous_class = corrections.streak, corrections.previous_class

            self.student.train()
            shuffled = list(train)
            random.shuffle(shuffled)
            train_loss = 0.0
            for group in batches(shuffled):
                ids = [graph.sample_id for graph in group]
                logits = self.student(gin, group, self.text_embeddings)
                if corrections is None:
                    loss = F.cross_entropy(logits, observed[ids])
                else:
                    loss = classification_loss(logits, observed[ids], corrections.weights[ids], corrections.soft_labels[ids])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite classification loss at epoch {epoch}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                update_teacher(self.student, teacher)
                train_loss += loss.item() * len(group)

            history_update = 0.0
            if corrections is not None:
                progress = (epoch - WARMUP_EPOCHS + 1) / (epochs - WARMUP_EPOCHS)
                history_update = self._refresh_history(groups, teacher, history, 0.05 + (0.20 - 0.05) * progress)
            correct = self._count_correct(test)
            record_epoch({
                "epoch": epoch, "train_loss": train_loss / len(train), "test_acc": correct / len(test),
                "test_correct": correct, "test_size": len(test), "prototype_counts": counts,
                "selected": corrections.selected.sum().item() if corrections is not None else 0,
                "mean_correction_weight": corrections.weights.mean().item() if corrections is not None else 0.0,
                "max_history_update": history_update,
            })
