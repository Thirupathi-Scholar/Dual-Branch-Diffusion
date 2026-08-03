"""
losses.py — composite objective for DualDiffSeg (manuscript Section 3.7).

    L_total = l1*L_seg + l2*L_diff + l3*L_conf + l4*L_MRI

Every term is returned separately as well as combined, so that training logs
record the contribution of each component. That log is what lets you say
something substantive about the loss-term ablations rather than only reporting
a final Dice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, eps=1e-5):
    """Soft Dice over the foreground channel."""
    p = torch.sigmoid(logits)
    p = p.flatten(1)
    g = target.flatten(1).float()
    inter = (p * g).sum(1)
    return (1.0 - (2.0 * inter + eps) / (p.sum(1) + g.sum(1) + eps)).mean()


def diffusion_consistency(traj):
    """
    L_diff = sum_t || F_t - F_{t+1} ||^2   (manuscript eq. 5)

    Penalizes abrupt change between consecutive refinement stages, which is
    what keeps the branch from amplifying high-frequency noise. Normalized by
    element count so the magnitude does not depend on patch size.
    """
    if traj is None or len(traj) < 2:
        return torch.tensor(0.0)
    total = 0.0
    for a, b in zip(traj[:-1], traj[1:]):
        total = total + F.mse_loss(a, b)
    return total / (len(traj) - 1)


def confidence_entropy(conf, eps=1e-6):
    """
    L_conf = -sum(C_i * log(C_i + eps))   (manuscript eq. 6)

    Minimizing entropy sharpens the fusion weights, pushing CAFU toward
    decisive per-voxel choices instead of a near-uniform average.
    """
    if conf is None:
        return torch.tensor(0.0)
    return -(conf * torch.log(conf + eps)).sum(dim=1).mean()


def intensity_homogeneity(logits, image, eps=1e-6):
    """
    L_hom = Var(I_tumor)   (manuscript eq. 7)

    Soft-mask weighted intensity variance inside the predicted tumor region.
    """
    p = torch.sigmoid(logits)
    w = p / (p.sum(dim=(1, 2, 3, 4), keepdim=True) + eps)
    mean = (w * image).sum(dim=(1, 2, 3, 4), keepdim=True)
    var = (w * (image - mean) ** 2).sum(dim=(1, 2, 3, 4))
    return var.mean()


class DualDiffSegLoss(nn.Module):
    """
    Composite loss. Set any lambda to 0.0 to run the corresponding loss-term
    ablation without touching the architecture.

    NOTE ON alpha AND THE LAMBDAS: the manuscript defines these symbols but
    never states their values. The defaults below are reasonable starting
    points, not values recovered from the original work. Whatever you use,
    record it — a reported result whose loss weights are unknown is not
    reproducible, and that is one of the gaps flagged in the review.
    """

    def __init__(self, alpha=0.5, lambda_seg=1.0, lambda_diff=0.1,
                 lambda_conf=0.05, lambda_mri=0.01, pos_weight=None,
                 conf_warmup=20, conf_ramp=30, conf_floor=0.05):
        """
        conf_warmup / conf_ramp / conf_floor control the confidence term's
        schedule. The entropy penalty has no opposing force, so any constant
        positive weight eventually drives the CAFU softmax to one-hot and the
        unit degenerates into a hard branch selector, defeating the adaptive
        weighting it exists to provide.

        Two safeguards:
          - the weight is held at zero for `conf_warmup` epochs, then ramped
            linearly to its full value over `conf_ramp` epochs, so the network
            learns WHERE each branch is reliable before being pushed to commit;
          - the penalty is switched off once mean entropy falls below
            `conf_floor`, which stops it collapsing further while leaving it
            free to re-engage if the maps soften again.

        Set conf_warmup=0 and conf_floor=0 to reproduce an unscheduled
        constant weight.
        """
        super().__init__()
        self.alpha = alpha
        self.l_seg = lambda_seg
        self.l_diff = lambda_diff
        self.l_conf = lambda_conf
        self.l_mri = lambda_mri
        self.conf_warmup = conf_warmup
        self.conf_ramp = max(1, conf_ramp)
        self.conf_floor = conf_floor
        self.epoch = 0
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight) if pos_weight is not None else None,
        )

    def set_epoch(self, epoch):
        """Called once per epoch by the training loop to advance the schedule."""
        self.epoch = int(epoch)

    def current_conf_weight(self, entropy=None):
        """Effective lambda_conf for the current epoch."""
        if self.l_conf <= 0:
            return 0.0
        if self.epoch < self.conf_warmup:
            return 0.0
        # entropy has already collapsed — stop pushing
        if entropy is not None and self.conf_floor > 0 and entropy < self.conf_floor:
            return 0.0
        progress = min(1.0, (self.epoch - self.conf_warmup + 1) / self.conf_ramp)
        return self.l_conf * progress

    def forward(self, out, target, image=None):
        logits = out["logits"]
        dev = logits.device

        l_dice = dice_loss(logits, target)
        l_ce = F.binary_cross_entropy_with_logits(
            logits, target.float(),
            pos_weight=self.pos_weight.to(dev) if self.pos_weight is not None else None,
        )
        l_seg = self.alpha * l_dice + (1.0 - self.alpha) * l_ce

        l_diff = diffusion_consistency(out.get("diff_traj")).to(dev)
        l_conf = confidence_entropy(out.get("conf")).to(dev)
        l_mri = (intensity_homogeneity(logits, image).to(dev)
                 if (image is not None and self.l_mri > 0)
                 else torch.tensor(0.0, device=dev))

        w_conf = self.current_conf_weight(entropy=float(l_conf.detach()))

        total = (self.l_seg * l_seg + self.l_diff * l_diff
                 + w_conf * l_conf + self.l_mri * l_mri)

        return total, dict(
            total=float(total.detach()), seg=float(l_seg.detach()),
            dice=float(l_dice.detach()), ce=float(l_ce.detach()),
            diff=float(l_diff.detach()), conf=float(l_conf.detach()),
            mri=float(l_mri.detach()), w_conf=w_conf,
        )
