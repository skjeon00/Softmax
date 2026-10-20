"""Core reliability calculations used in the manuscript.

Model interface:
    logits, h = model(x)
    logits: [B, C]
    h: [B, D, L]
    model.classifier.weight: [C, D]

The model uses temporal mean pooling and a bias-free linear classifier.
Use the trained model in evaluation mode: model.eval().
Larger scores indicate higher estimated reliability.
"""

import math

import torch


EPS = 1e-8

NON_AUG_COMPONENTS = (
    "Margin",
    "TemporalAgreement",
    "PairwiseConsistency",
    "LocalCertainty",
)


# ============================================================
# 1. LOCAL CLASSIFIER EVIDENCE
# ============================================================

def raw_local_evidence(h, classifier_weight):
    """h: [B, D, L], weight: [C, D] -> evidence: [B, L, C]."""
    return h.transpose(1, 2) @ classifier_weight.T


# ============================================================
# 2. ENTROPY AND AUGMENTATION CONSISTENCY
# ============================================================

def normalized_entropy(prob):
    """Entropy normalized by log(number of classes)."""
    entropy = -(prob * prob.clamp_min(EPS).log()).sum(dim=-1)
    return entropy / math.log(prob.shape[-1])


def weak_augmentations(x):
    """Three deterministic transformations used for consistency."""
    return [
        x * 0.98,
        x * 1.02,
        torch.roll(x, shifts=1, dims=-1),
    ]


@torch.no_grad()
def augmentation_consistency(model, x, base_prob):
    """exp(-mean symmetric KL divergence across augmentations)."""
    log_base = base_prob.clamp_min(EPS).log()
    divergences = []

    for x_aug in weak_augmentations(x):
        aug_logits, _ = model(x_aug)
        aug_prob = torch.softmax(aug_logits, dim=1)
        log_aug = aug_prob.clamp_min(EPS).log()

        kl_base_aug = (base_prob * (log_base - log_aug)).sum(dim=1)
        kl_aug_base = (aug_prob * (log_aug - log_base)).sum(dim=1)

        divergences.append(0.5 * (kl_base_aug + kl_aug_base))

    mean_divergence = torch.stack(divergences, dim=1).mean(dim=1)
    return torch.exp(-mean_divergence)


# ============================================================
# 3. ELEMENTARY RELIABILITY SCORES
# ============================================================

@torch.no_grad()
def compute_reliability_features(model, x, include_augmentation=True):
    """Return logits, probabilities, and elementary scores [B]."""
    logits, h = model(x)
    prob = torch.softmax(logits, dim=1)
    evidence = raw_local_evidence(h, model.classifier.weight)

    top2_values, top2_indices = logits.topk(k=2, dim=1)
    top1_value = top2_values[:, 0]
    top2_value = top2_values[:, 1]
    top1_class = top2_indices[:, 0]

    # Local evidence for the global top-1 and top-2 classes.
    local_top2 = evidence.gather(
        dim=2,
        index=top2_indices[:, None, :].expand(
            -1, evidence.shape[1], -1
        ),
    )

    local_prob = torch.softmax(evidence, dim=2)

    features = {
        # Conventional output-level confidence.
        "MSP": prob.max(dim=1).values,

        "EntropyConfidence": 1.0 - normalized_entropy(prob),

        "Margin": (top1_value - top2_value) / (
            top1_value.abs() + top2_value.abs() + EPS
        ),

        # Internal temporal evidence.
        "TemporalAgreement": (
            evidence.argmax(dim=2) == top1_class[:, None]
        ).float().mean(dim=1),

        "PairwiseConsistency": (
            local_top2[:, :, 0] > local_top2[:, :, 1]
        ).float().mean(dim=1),

        "LocalCertainty": (
            1.0 - normalized_entropy(local_prob).mean(dim=1)
        ),
    }

    if include_augmentation:
        features["AugmentationConsistency"] = augmentation_consistency(
            model, x, prob
        )

    return logits, prob, features


# ============================================================
# 4. CLEAN-TRAINING REFERENCE STATISTICS
# ============================================================

@torch.no_grad()
def compute_training_statistics(train_features):
    """Each value contains scores for all clean training samples.

    Calculate these statistics once per trained model/run.
    Reuse the same statistics for clean and corrupted evaluation.
    """
    components = list(NON_AUG_COMPONENTS)

    if "AugmentationConsistency" in train_features:
        components.append("AugmentationConsistency")

    train_mean = {
        name: train_features[name].mean()
        for name in components
    }

    train_std = {
        name: train_features[name].std(unbiased=False)
        for name in components
    }

    return train_mean, train_std


# ============================================================
# 5. COMPOSITE RELIABILITY SCORES
# ============================================================

@torch.no_grad()
def add_composite_scores(features, train_mean, train_std):
    """Average standardized components using clean-training statistics."""
    scores = dict(features)
    components = list(NON_AUG_COMPONENTS)

    if "AugmentationConsistency" in features:
        components.append("AugmentationConsistency")

    standardized = {
        name: (features[name] - train_mean[name]) / (
            train_std[name] + EPS
        )
        for name in components
    }

    # Margin + temporal agreement + pairwise consistency + local certainty.
    scores["NonAugmentationComposite"] = torch.stack(
        [standardized[name] for name in NON_AUG_COMPONENTS],
        dim=1,
    ).mean(dim=1)

    # The same four components plus augmentation consistency.
    if "AugmentationConsistency" in standardized:
        scores["FullComposite"] = torch.stack(
            [standardized[name] for name in components],
            dim=1,
        ).mean(dim=1)

    return scores