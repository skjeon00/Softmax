# ============================================================
# Systematic Reliability Benchmark for Wearable HAR
# UCI-HAR: Clean + Deployment-Relevant Corruptions
# ============================================================
#
# RESEARCH QUESTION
# -----------------
# When is simple softmax confidence sufficient for trustworthy
# wearable HAR, and when does it fail under signal degradation?
#
#
# DATASET
# -------
# UCI-HAR official train/test split
#
#
# BACKBONES
# ---------
# 1. CNN
# 2. BiLSTM
# 3. Transformer
#
# Every backbone returns temporal features h[t] and uses:
#
#       mean pooling -> bias-free linear classifier
#
# This enables a controlled comparison of reliability behavior.
#
#
# RELIABILITY METHODS
# -------------------
# Conventional confidence:
#   - MSP
#   - Predictive entropy
#   - Top1-top2 margin
#
# Internal temporal evidence:
#   - Temporal agreement
#   - Pairwise consistency
#   - Local certainty
#
# Consistency-based:
#   - Augmentation consistency
#
# Combined:
#   - Internal evidence only
#   - Full reliability
#
#
# EVALUATION CONDITIONS
# ---------------------
# Clean
#
# Additive Gaussian noise:
#   sigma = 0.05, 0.10, 0.20
#
# Additive bias drift:
#   delta = 0.05, 0.10, 0.20
#
# Multiplicative scale drift:
#   beta = -0.20, -0.10, +0.10, +0.20
#
# Temporal scaling:
#   s = 0.85, 0.90, 0.95, 1.05, 1.10, 1.15
#
#
# PRIMARY METRICS
# ---------------
# Classification:
#   - Accuracy
#   - Macro-F1
#
# Error detection:
#   - AUROC
#   - AUPRC
#
# Calibration:
#   - ECE
#   - Brier score
#   - NLL
#
# Selective prediction:
#   - Risk-coverage
#   - AURC
#
# Deployment:
#   - CPU latency
#
#
# IMPORTANT
# ---------
# - Input standardization uses TRAIN only.
# - Combined reliability feature normalization uses TRAIN only.
# - Test labels are NEVER used to define reliability scores.
# - Corruptions are applied only at evaluation time.
#
#
# EXPECTED DIRECTORY
# ------------------
# ./UCI HAR Dataset/
#     train/
#       Inertial Signals/
#       y_train.txt
#     test/
#       Inertial Signals/
#       y_test.txt
#
#
# REQUIREMENTS
# ------------
# pip install numpy pandas scikit-learn torch
# ============================================================

import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader




# ============================================================
# 0. CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 6
INPUT_CHANNELS = 9
INPUT_LENGTH = 128
FEATURE_DIM = 128

BATCH_SIZE = 128
EPOCHS = 35
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EPS = 1e-8

CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

# Risk-coverage
COVERAGES = [
    1.00,
    0.95,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
]

# Calibration
ECE_BINS = 15

# Runtime
CPU_LATENCY_WARMUP = 30
CPU_LATENCY_RUNS = 150

# Reliability feature sets
INTERNAL_FEATURES = [
    "Margin",
    "TemporalAgreement",
    "PairwiseConsistency",
    "LocalCertainty",
]

FULL_FEATURES = (
    INTERNAL_FEATURES
    + [
        "AugmentationConsistency"
    ]
)

# Corruption settings.
# Values operate on TRAIN-standardized input.
CORRUPTION_SETTINGS = [
    ("clean", 0.0),

    ("gaussian_noise", 0.05),
    ("gaussian_noise", 0.10),
    ("gaussian_noise", 0.20),

    ("bias_drift", 0.05),
    ("bias_drift", 0.10),
    ("bias_drift", 0.20),

    ("scale_drift", -0.20),
    ("scale_drift", -0.10),
    ("scale_drift", 0.10),
    ("scale_drift", 0.20),

    ("temporal_scaling", 0.85),
    ("temporal_scaling", 0.90),
    ("temporal_scaling", 0.95),
    ("temporal_scaling", 1.05),
    ("temporal_scaling", 1.10),
    ("temporal_scaling", 1.15),
]




# ============================================================
# 4. COMMON MODEL INTERFACE
# ============================================================

class HARBase(nn.Module):
    """
    Every model must return:
        logits: [B,C]
        h:      [B,D,T_local]

    The final classifier is bias-free:
        mean_t(W h_t) == W mean_t(h_t)
    """

    def classify_temporal_features(
        self,
        h,
    ):
        pooled = h.mean(
            dim=-1
        )

        logits = self.classifier(
            pooled
        )

        return logits




# ============================================================
# 7. TRANSFORMER
# ============================================================

class TransformerHAR(HARBase):
    def __init__(
        self,
    ):
        super().__init__()

        self.input_proj = nn.Linear(
            INPUT_CHANNELS,
            FEATURE_DIM,
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                INPUT_LENGTH,
                FEATURE_DIM,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FEATURE_DIM,
            nhead=4,
            dim_feedforward=FEATURE_DIM * 2,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.norm = nn.LayerNorm(
            FEATURE_DIM
        )

        self.classifier = nn.Linear(
            FEATURE_DIM,
            NUM_CLASSES,
            bias=False,
        )

        nn.init.normal_(
            self.positional_embedding,
            mean=0.0,
            std=0.02,
        )

    def extract_local(
        self,
        x,
    ):
        # [B,C,T] -> [B,T,C]
        z = x.transpose(
            1,
            2,
        )

        z = self.input_proj(
            z
        )

        z = (
            z
            + self.positional_embedding[
                :,
                :z.shape[1],
                :,
            ]
        )

        z = self.encoder(
            z
        )

        z = self.norm(
            z
        )

        return z.transpose(
            1,
            2,
        )

    def forward(
        self,
        x,
    ):
        h = self.extract_local(
            x
        )

        logits = (
            self.classify_temporal_features(
                h
            )
        )

        return (
            logits,
            h,
        )


def build_model(
    backbone_name,
):
    if backbone_name == "CNN":
        return CNNHAR()

    if backbone_name == "BiLSTM":
        return BiLSTMHAR()

    if backbone_name == "Transformer":
        return TransformerHAR()

    raise ValueError(
        f"Unknown backbone: {backbone_name}"
    )




# ============================================================
# 8. TRAINING
# ============================================================

def train_model(
    model,
    loader,
):
    model = model.to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_loss = float(
        "inf"
    )

    best_state = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_n = 0

        for xb, yb in loader:
            xb = xb.to(
                DEVICE
            )

            yb = yb.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, _ = model(
                xb
            )

            loss = F.cross_entropy(
                logits,
                yb,
            )

            loss.backward()
            optimizer.step()

            bs = xb.size(
                0
            )

            total_loss += (
                loss.item()
                * bs
            )

            total_correct += (
                logits.argmax(
                    dim=1
                )
                == yb
            ).sum().item()

            total_n += bs

        mean_loss = (
            total_loss
            / total_n
        )

        train_acc = (
            total_correct
            / total_n
        )

        if mean_loss < best_loss:
            best_loss = mean_loss

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):
            print(
                f"epoch={epoch:02d}/{EPOCHS} | "
                f"loss={mean_loss:.4f} | "
                f"train_acc={train_acc:.4f}"
            )

    model.load_state_dict(
        best_state
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model




# ============================================================
# 9. CORRUPTIONS
# ============================================================

def temporal_scale_batch(
    x,
    scale,
):
    """
    Simulate execution-speed changes.

    x: [B,C,T]

    1) resample to round(T * scale)
    2) center-crop or edge-pad back to T

    scale < 1:
        shorter/faster temporal pattern

    scale > 1:
        longer/slower temporal pattern
    """
    B, C, T = x.shape

    new_length = max(
        2,
        int(
            round(
                T * scale
            )
        ),
    )

    resized = F.interpolate(
        x,
        size=new_length,
        mode="linear",
        align_corners=False,
    )

    if new_length == T:
        return resized

    if new_length > T:
        start = (
            new_length - T
        ) // 2

        return resized[
            :,
            :,
            start:start + T,
        ]

    # Pad to T using edge replication.
    total_pad = (
        T - new_length
    )

    left = (
        total_pad // 2
    )

    right = (
        total_pad - left
    )

    return F.pad(
        resized,
        pad=(
            left,
            right,
        ),
        mode="replicate",
    )


def apply_corruption(
    x,
    corruption_name,
    severity,
    noise_seed=12345,
):
    """
    Corruptions operate on standardized input.

    Deterministic noise is used for reproducibility.
    """
    if corruption_name == "clean":
        return x

    if corruption_name == "gaussian_noise":
        generator = torch.Generator(
            device=x.device
        )

        generator.manual_seed(
            noise_seed
        )

        noise = torch.randn(
            x.shape,
            generator=generator,
            device=x.device,
            dtype=x.dtype,
        )

        return (
            x
            + severity * noise
        )

    if corruption_name == "bias_drift":
        return (
            x
            + severity
        )

    if corruption_name == "scale_drift":
        return (
            x
            * (
                1.0
                + severity
            )
        )

    if corruption_name == "temporal_scaling":
        return temporal_scale_batch(
            x,
            severity,
        )

    raise ValueError(
        f"Unknown corruption: {corruption_name}"
    )


def condition_label(
    name,
    severity,
):
    if name == "clean":
        return "clean"

    return (
        f"{name}:{severity:+.2f}"
    )




# ============================================================
# 10. LOCAL CLASSIFIER EVIDENCE
# ============================================================

def raw_local_evidence(
    h,
    classifier_weight,
):
    """
    h: [B,D,T]
    W: [C,D]

    Returns:
        evidence [B,T,C]
    """
    z = h.transpose(
        1,
        2,
    )

    return torch.einsum(
        "btd,cd->btc",
        z,
        classifier_weight,
    )




# ============================================================
# 11. RELIABILITY UTILITIES
# ============================================================

def normalized_entropy(
    prob,
):
    entropy = -(
        prob
        * torch.log(
            torch.clamp(
                prob,
                min=EPS,
            )
        )
    ).sum(
        dim=-1
    )

    return (
        entropy
        / math.log(
            prob.shape[-1]
        )
    )


def weak_augmentations(
    x,
):
    """
    Weak transformations used only for consistency estimation.
    """
    return [
        x * 0.98,
        x * 1.02,
        torch.roll(
            x,
            shifts=1,
            dims=-1,
        ),
    ]


@torch.no_grad()
def augmentation_consistency(
    model,
    x,
    base_prob,
):
    divergences = []

    for x_aug in weak_augmentations(
        x
    ):
        aug_logits, _ = model(
            x_aug
        )

        aug_prob = torch.softmax(
            aug_logits,
            dim=1,
        )

        kl_pa = (
            base_prob
            * (
                torch.log(
                    torch.clamp(
                        base_prob,
                        min=EPS,
                    )
                )
                - torch.log(
                    torch.clamp(
                        aug_prob,
                        min=EPS,
                    )
                )
            )
        ).sum(
            dim=1
        )

        kl_ap = (
            aug_prob
            * (
                torch.log(
                    torch.clamp(
                        aug_prob,
                        min=EPS,
                    )
                )
                - torch.log(
                    torch.clamp(
                        base_prob,
                        min=EPS,
                    )
                )
            )
        ).sum(
            dim=1
        )

        divergences.append(
            0.5
            * (
                kl_pa
                + kl_ap
            )
        )

    mean_divergence = torch.stack(
        divergences,
        dim=1,
    ).mean(
        dim=1
    )

    return torch.exp(
        -mean_divergence
    )


@torch.no_grad()
def compute_reliability_features(
    model,
    x,
    include_augmentation=True,
):
    logits, h = model(
        x
    )

    prob = torch.softmax(
        logits,
        dim=1,
    )

    W = (
        model.classifier
        .weight.detach()
    )

    evidence = raw_local_evidence(
        h,
        W,
    )

    # --------------------------------------------------------
    # conventional softmax confidence
    # --------------------------------------------------------
    msp = prob.max(
        dim=1
    ).values

    entropy_confidence = (
        1.0
        - normalized_entropy(
            prob
        )
    )

    top2 = torch.topk(
        logits,
        k=2,
        dim=1,
    )

    top1_value = top2.values[
        :,
        0
    ]

    top2_value = top2.values[
        :,
        1
    ]

    top1_idx = top2.indices[
        :,
        0
    ]

    top2_idx = top2.indices[
        :,
        1
    ]

    margin = (
        top1_value
        - top2_value
    ) / (
        top1_value.abs()
        + top2_value.abs()
        + EPS
    )

    # --------------------------------------------------------
    # temporal agreement
    # --------------------------------------------------------
    local_pred = evidence.argmax(
        dim=2
    )

    temporal_agreement = (
        local_pred
        == top1_idx[:, None]
    ).float().mean(
        dim=1
    )

    # --------------------------------------------------------
    # pairwise top1-vs-top2 temporal consistency
    # --------------------------------------------------------
    B, T, _ = evidence.shape

    b_idx = torch.arange(
        B,
        device=x.device,
    )

    t_idx = torch.arange(
        T,
        device=x.device,
    )

    top1_local = evidence[
        b_idx[:, None],
        t_idx[None, :],
        top1_idx[:, None],
    ]

    top2_local = evidence[
        b_idx[:, None],
        t_idx[None, :],
        top2_idx[:, None],
    ]

    pairwise_consistency = (
        top1_local
        > top2_local
    ).float().mean(
        dim=1
    )

    # --------------------------------------------------------
    # local certainty
    # --------------------------------------------------------
    local_prob = torch.softmax(
        evidence,
        dim=2,
    )

    local_certainty = (
        1.0
        - normalized_entropy(
            local_prob
        ).mean(
            dim=1
        )
    )

    # --------------------------------------------------------
    # augmentation consistency
    # --------------------------------------------------------
    if include_augmentation:
        aug_consistency = (
            augmentation_consistency(
                model,
                x,
                prob,
            )
        )
    else:
        aug_consistency = torch.ones(
            B,
            device=x.device,
            dtype=x.dtype,
        )

    features = {
        "MSP": msp,
        "EntropyConfidence": (
            entropy_confidence
        ),
        "Margin": margin,
        "TemporalAgreement": (
            temporal_agreement
        ),
        "PairwiseConsistency": (
            pairwise_consistency
        ),
        "LocalCertainty": (
            local_certainty
        ),
        "AugmentationConsistency": (
            aug_consistency
        ),
    }

    # Exact classifier/evidence equivalence check.
    uniform_scores = evidence.mean(
        dim=1
    )

    equivalence_error = (
        logits
        - uniform_scores
    ).abs().max()

    return (
        logits,
        prob,
        features,
        equivalence_error,
    )




# ============================================================
# 12. COLLECT TRAIN FEATURES FOR SCORE STANDARDIZATION
# ============================================================

@torch.no_grad()
def collect_clean_train_features(
    model,
    loader,
):
    model.eval()

    buffers = {
        "MSP": [],
        "EntropyConfidence": [],
        "Margin": [],
        "TemporalAgreement": [],
        "PairwiseConsistency": [],
        "LocalCertainty": [],
        "AugmentationConsistency": [],
    }

    max_equivalence_error = 0.0

    for xb, _ in loader:
        xb = xb.to(
            DEVICE
        )

        (
            _,
            _,
            features,
            eq_error,
        ) = (
            compute_reliability_features(
                model,
                xb,
                include_augmentation=True,
            )
        )

        max_equivalence_error = max(
            max_equivalence_error,
            eq_error.item(),
        )

        for name in buffers:
            buffers[
                name
            ].append(
                features[
                    name
                ].cpu()
            )

    features_np = {
        name: torch.cat(
            chunks
        ).numpy()
        for name, chunks
        in buffers.items()
    }

    return (
        features_np,
        max_equivalence_error,
    )


def fit_feature_stats(
    train_features,
):
    stats = {}

    for name in FULL_FEATURES:
        values = np.asarray(
            train_features[
                name
            ],
            dtype=np.float64,
        )

        stats[
            name
        ] = {
            "mean": float(
                values.mean()
            ),
            "std": float(
                values.std()
                + EPS
            ),
        }

    return stats


def combine_features(
    features,
    stats,
    names,
):
    z_list = []

    for name in names:
        values = np.asarray(
            features[
                name
            ],
            dtype=np.float64,
        )

        mean = stats[
            name
        ][
            "mean"
        ]

        std = stats[
            name
        ][
            "std"
        ]

        z = (
            values
            - mean
        ) / std

        z_list.append(
            z
        )

    return np.stack(
        z_list,
        axis=1,
    ).mean(
        axis=1
    )


def build_reliability_scores(
    features,
    stats,
):
    return {
        "MSP": (
            features[
                "MSP"
            ]
        ),

        "Predictive entropy": (
            features[
                "EntropyConfidence"
            ]
        ),

        "Top1-top2 margin": (
            features[
                "Margin"
            ]
        ),

        "Temporal agreement": (
            features[
                "TemporalAgreement"
            ]
        ),

        "Pairwise consistency": (
            features[
                "PairwiseConsistency"
            ]
        ),

        "Local certainty": (
            features[
                "LocalCertainty"
            ]
        ),

        "Augmentation consistency": (
            features[
                "AugmentationConsistency"
            ]
        ),

        "Internal evidence only": (
            combine_features(
                features,
                stats,
                INTERNAL_FEATURES,
            )
        ),

        "Full reliability": (
            combine_features(
                features,
                stats,
                FULL_FEATURES,
            )
        ),
    }




# ============================================================
# 13. EVALUATE ONE CONDITION
# ============================================================

@torch.no_grad()
def evaluate_condition(
    model,
    loader,
    feature_stats,
    corruption_name,
    severity,
    seed,
):
    model.eval()

    y_all = []
    logits_all = []
    prob_all = []

    feature_buffers = {
        "MSP": [],
        "EntropyConfidence": [],
        "Margin": [],
        "TemporalAgreement": [],
        "PairwiseConsistency": [],
        "LocalCertainty": [],
        "AugmentationConsistency": [],
    }

    max_equivalence_error = 0.0

    batch_index = 0

    for xb, yb in loader:
        xb = xb.to(
            DEVICE
        )

        # Deterministic but batch-distinct noise seed.
        noise_seed = (
            100000
            + seed * 1000
            + batch_index
        )

        x_eval = apply_corruption(
            xb,
            corruption_name,
            severity,
            noise_seed=noise_seed,
        )

        (
            logits,
            prob,
            features,
            eq_error,
        ) = (
            compute_reliability_features(
                model,
                x_eval,
                include_augmentation=True,
            )
        )

        max_equivalence_error = max(
            max_equivalence_error,
            eq_error.item(),
        )

        y_all.append(
            yb.cpu()
        )

        logits_all.append(
            logits.cpu()
        )

        prob_all.append(
            prob.cpu()
        )

        for name in feature_buffers:
            feature_buffers[
                name
            ].append(
                features[
                    name
                ].cpu()
            )

        batch_index += 1

    y = torch.cat(
        y_all
    ).numpy()

    logits = torch.cat(
        logits_all
    ).numpy()

    prob = torch.cat(
        prob_all
    ).numpy()

    features_np = {
        name: torch.cat(
            chunks
        ).numpy()
        for name, chunks
        in feature_buffers.items()
    }

    reliability_scores = (
        build_reliability_scores(
            features_np,
            feature_stats,
        )
    )

    return {
        "y": y,
        "logits": logits,
        "prob": prob,
        "pred": logits.argmax(
            axis=1
        ),
        "features": features_np,
        "reliability_scores": reliability_scores,
        "max_equivalence_error": max_equivalence_error,
    }




# ============================================================
# 14. CLASSIFICATION + CALIBRATION
# ============================================================

def expected_calibration_error(
    y,
    prob,
    n_bins=ECE_BINS,
):
    confidence = prob.max(
        axis=1
    )

    pred = prob.argmax(
        axis=1
    )

    correct = (
        pred == y
    ).astype(
        np.float64
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(
        n_bins
    ):
        lo = bin_edges[
            i
        ]

        hi = bin_edges[
            i + 1
        ]

        if i == n_bins - 1:
            mask = (
                (confidence >= lo)
                & (confidence <= hi)
            )
        else:
            mask = (
                (confidence >= lo)
                & (confidence < hi)
            )

        if mask.sum() == 0:
            continue

        bin_acc = correct[
            mask
        ].mean()

        bin_conf = confidence[
            mask
        ].mean()

        ece += (
            mask.mean()
            * abs(
                bin_acc
                - bin_conf
            )
        )

    return float(
        ece
    )


def brier_multiclass(
    y,
    prob,
):
    target = np.eye(
        NUM_CLASSES,
        dtype=np.float64,
    )[
        y
    ]

    return float(
        np.mean(
            np.sum(
                (
                    prob
                    - target
                ) ** 2,
                axis=1,
            )
        )
    )


def calibration_and_classification_row(
    seed,
    backbone,
    condition,
    result,
):
    y = result[
        "y"
    ]

    pred = result[
        "pred"
    ]

    prob = result[
        "prob"
    ]

    return {
        "Seed": seed,
        "Backbone": backbone,
        "Condition": condition,

        "Accuracy": accuracy_score(
            y,
            pred,
        ),

        "Macro_F1": f1_score(
            y,
            pred,
            average="macro",
            zero_division=0,
        ),

        "Error_Rate": (
            1.0
            - accuracy_score(
                y,
                pred,
            )
        ),

        "ECE": expected_calibration_error(
            y,
            prob,
        ),

        "Brier": brier_multiclass(
            y,
            prob,
        ),

        "NLL": log_loss(
            y,
            prob,
            labels=np.arange(
                NUM_CLASSES
            ),
        ),

        "MaxAbs_LogitDifference_Standard_vs_Uniform": (
            result[
                "max_equivalence_error"
            ]
        ),
    }




# ============================================================
# 15. ERROR DETECTION
# ============================================================

def error_detection_rows(
    seed,
    backbone,
    condition,
    result,
):
    y = result[
        "y"
    ]

    pred = result[
        "pred"
    ]

    error = (
        pred != y
    ).astype(
        np.int64
    )

    rows = []

    for method, reliability in (
        result[
            "reliability_scores"
        ].items()
    ):
        error_score = (
            -np.asarray(
                reliability
            )
        )

        if len(
            np.unique(
                error
            )
        ) == 2:
            auroc = roc_auc_score(
                error,
                error_score,
            )

            auprc = average_precision_score(
                error,
                error_score,
            )
        else:
            auroc = float(
                "nan"
            )

            auprc = float(
                "nan"
            )

        rows.append(
            {
                "Seed": seed,
                "Backbone": backbone,
                "Condition": condition,
                "Method": method,
                "Error_Detection_AUROC": (
                    auroc
                ),
                "Error_Detection_AUPRC": (
                    auprc
                ),
            }
        )

    return rows




# ============================================================
# 16. RISK-COVERAGE / AURC
# ============================================================

def compute_aurc(
    y,
    pred,
    reliability,
):
    order = np.argsort(
        -np.asarray(
            reliability
        )
    )

    errors = (
        pred[
            order
        ]
        != y[
            order
        ]
    ).astype(
        np.float64
    )

    cumulative_error = np.cumsum(
        errors
    )

    k = np.arange(
        1,
        len(errors) + 1,
        dtype=np.float64,
    )

    risk = (
        cumulative_error
        / k
    )

    return float(
        risk.mean()
    )


def risk_coverage_rows(
    seed,
    backbone,
    condition,
    result,
):
    y = result[
        "y"
    ]

    pred = result[
        "pred"
    ]

    rows = []
    aurc_rows = []

    for method, reliability in (
        result[
            "reliability_scores"
        ].items()
    ):
        reliability = np.asarray(
            reliability
        )

        order = np.argsort(
            -reliability
        )

        y_sorted = y[
            order
        ]

        pred_sorted = pred[
            order
        ]

        for coverage in COVERAGES:
            n_keep = max(
                1,
                int(
                    round(
                        coverage
                        * len(y)
                    )
                ),
            )

            y_keep = y_sorted[
                :n_keep
            ]

            pred_keep = pred_sorted[
                :n_keep
            ]

            accuracy = (
                pred_keep
                == y_keep
            ).mean()

            macro_f1 = f1_score(
                y_keep,
                pred_keep,
                average="macro",
                zero_division=0,
            )

            rows.append(
                {
                    "Seed": seed,
                    "Backbone": backbone,
                    "Condition": condition,
                    "Method": method,
                    "Coverage": coverage,
                    "Accuracy": accuracy,
                    "Selective_Risk": (
                        1.0
                        - accuracy
                    ),
                    "Macro_F1": macro_f1,
                }
            )

        aurc_rows.append(
            {
                "Seed": seed,
                "Backbone": backbone,
                "Condition": condition,
                "Method": method,
                "AURC": compute_aurc(
                    y,
                    pred,
                    reliability,
                ),
            }
        )

    return (
        rows,
        aurc_rows,
    )




# ============================================================
# 17. CPU LATENCY
# ============================================================

@torch.no_grad()
def cpu_latency_benchmark(
    model,
    sample,
):
    model_cpu = copy.deepcopy(
        model
    ).cpu().eval()

    x = sample[
        :1
    ].cpu()

    def base_fn():
        model_cpu(
            x
        )

    def internal_fn():
        compute_reliability_features(
            model_cpu,
            x,
            include_augmentation=False,
        )

    def full_fn():
        compute_reliability_features(
            model_cpu,
            x,
            include_augmentation=True,
        )

    for _ in range(
        CPU_LATENCY_WARMUP
    ):
        base_fn()
        internal_fn()
        full_fn()

    def measure(
        fn,
    ):
        start = time.perf_counter()

        for _ in range(
            CPU_LATENCY_RUNS
        ):
            fn()

        return (
            time.perf_counter()
            - start
        ) * 1000.0 / CPU_LATENCY_RUNS

    base_ms = measure(
        base_fn
    )

    internal_ms = measure(
        internal_fn
    )

    full_ms = measure(
        full_fn
    )

    return {
        "Baseline_CPU_ms": (
            base_ms
        ),

        "InternalEvidence_CPU_ms": (
            internal_ms
        ),

        "FullReliability_CPU_ms": (
            full_ms
        ),

        "Internal_Overhead_pct": (
            100.0
            * (
                internal_ms
                - base_ms
            )
            / base_ms
        ),

        "Full_Overhead_pct": (
            100.0
            * (
                full_ms
                - base_ms
            )
            / base_ms
        ),
    }



