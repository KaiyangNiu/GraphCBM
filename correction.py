"""Fuse prototype evidence, confirm changes with the EMA teacher, and update soft labels."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ConfirmedCorrections:
    weights: torch.Tensor
    soft_labels: torch.Tensor
    streak: torch.Tensor
    previous_class: torch.Tensor
    selected: torch.Tensor


def initialize_soft_labels(observed, num_classes):
    return 0.9 * F.one_hot(observed, num_classes).float() + 0.1 / num_classes


@torch.no_grad()
def confirm_corrections(structure_prob, semantic_prob, teacher_prob, observed, history, streak, previous_class):
    candidate_prob = (structure_prob.clamp_min(1e-8).log() + semantic_prob.clamp_min(1e-8).log()).softmax(dim=1)
    confidence, candidate = candidate_prob.max(dim=1)
    str_confidence, str_class = structure_prob.max(dim=1)
    sem_confidence, sem_class = semantic_prob.max(dim=1)
    teacher_confidence, teacher_class = teacher_prob.max(dim=1)
    selected = (
        (candidate != observed) & (candidate == str_class) & (candidate == sem_class)
        & (candidate == teacher_class) & (confidence > 0.50) & (teacher_confidence > 0.70)
        & (str_confidence > 0.50) & (sem_confidence > 0.50)
    )
    candidates = selected.nonzero().flatten()
    budget = round(0.20 * len(observed))
    if len(candidates) > budget:
        scores = confidence[candidates] * teacher_confidence[candidates]
        keep = candidates[scores.topk(budget).indices]
        selected = torch.zeros_like(selected)
        selected[keep] = True

    next_streak = torch.where(selected, torch.where(candidate == previous_class, streak + 1, 1), 0)
    next_class = torch.where(selected, candidate, -1)
    continuity = torch.where(next_streak >= 2, 1.0, 0.1)
    weights = selected.to(candidate_prob.dtype) * (confidence * teacher_confidence).clamp(0.0, 1.0) * continuity
    soft_labels = (1.0 - weights[:, None]) * history + weights[:, None] * candidate_prob
    return ConfirmedCorrections(weights, soft_labels, next_streak, next_class, selected)


def classification_loss(logits, observed, weights, soft_labels):
    hard_loss = F.cross_entropy(logits, observed, reduction="none")
    soft_loss = -(soft_labels.detach() * logits.log_softmax(dim=-1)).sum(dim=-1)
    return ((1.0 - weights) * hard_loss + weights * soft_loss).mean()


@torch.no_grad()
def update_teacher(student, teacher, decay=0.99):
    for source, target in zip(student.parameters(), teacher.parameters()):
        target.mul_(decay).add_(source, alpha=1.0 - decay)
    for source, target in zip(student.buffers(), teacher.buffers()):
        if target.is_floating_point():
            target.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            target.copy_(source)


@torch.no_grad()
def update_soft_label_history(history, original_view, dropout_view, max_step):
    target = 0.5 * (original_view + dropout_view)
    log_mean = target.clamp_min(1e-8).log()
    kl_a = (original_view * (original_view.clamp_min(1e-8).log() - log_mean)).sum(dim=-1)
    kl_b = (dropout_view * (dropout_view.clamp_min(1e-8).log() - log_mean)).sum(dim=-1)
    agreement = (1.0 - 0.5 * (kl_a + kl_b) / math.log(2.0)).clamp(0.0, 1.0)
    entropy = -(target * log_mean).sum(dim=-1)
    certainty = (1.0 - entropy / math.log(target.size(-1))).clamp(0.0, 1.0)
    step = (max_step * (agreement * certainty)).clamp(0.0, 1.0)[:, None]
    updated = (1.0 - step) * history + step * target
    return updated / updated.sum(dim=-1, keepdim=True).clamp_min(1e-8)
