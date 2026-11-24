"""Build and score weighted prototypes independently within each observed class."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ClassPrototypes:
    centers: torch.Tensor
    weights: torch.Tensor
    members: list[list[int]]


def clustering_threshold(embeddings):
    if len(embeddings) < 3:
        return 0.80
    similarities = embeddings @ embeddings.T
    similarities.fill_diagonal_(float("-inf"))
    return torch.quantile(similarities.max(dim=1).values, 0.20).clamp(0.75, 0.90).item()


@torch.no_grad()
def build_prototype_bank(embeddings, observed, reliability, num_classes):
    # The reference normalizes at the scoring entry and again for clustering.
    normalized = F.normalize(F.normalize(embeddings.float(), dim=1), dim=1)
    reliability = reliability.to(normalized).clamp_min(1e-8)
    bank = []
    for class_id in range(num_classes):
        indices = (observed == class_id).nonzero().flatten()
        if not len(indices):
            bank.append(ClassPrototypes(normalized[:0], normalized.new_empty(0), []))
            continue

        threshold = clustering_threshold(normalized[indices])
        order = sorted(indices.tolist(), key=lambda index: (-float(reliability[index]), index))
        centers, members = [], []
        for index in order:
            embedding = normalized[index]
            if centers:
                similarity, nearest = (torch.stack(centers) @ embedding).max(dim=0)
                if similarity.item() >= threshold:
                    match = nearest.item()
                    group = members[match]
                    group.append(index)
                    weights = reliability[group, None]
                    average = (normalized[group] * weights).sum(dim=0) / weights.sum()
                    centers[match] = F.normalize(average, dim=0)
                    continue
            centers.append(embedding)
            members.append([index])

        masses = []
        for center, group in zip(centers, members):
            weights = reliability[group]
            effective_support = weights.sum().item()
            compactness = ((normalized[group] @ center) * weights).sum().item() / max(effective_support, 1e-8)
            masses.append(effective_support * max(compactness, 1e-8))
        masses = normalized.new_tensor(masses)
        bank.append(ClassPrototypes(torch.stack(centers), masses / masses.sum(), members))
    return bank


@torch.no_grad()
def score_prototypes(embeddings, bank, temperature=0.10):
    normalized = F.normalize(embeddings.float(), dim=1)
    logits = normalized.new_full((len(normalized), len(bank)), -1e6)
    for class_id, prototypes in enumerate(bank):
        if len(prototypes.centers):
            similarities = normalized @ prototypes.centers.T
            logits[:, class_id] = temperature * torch.logsumexp(
                similarities / temperature + prototypes.weights.clamp_min(1e-8).log(), dim=1,
            )
    return (logits / temperature).softmax(dim=1)
