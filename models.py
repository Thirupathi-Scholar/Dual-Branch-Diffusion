"""
models.py — DualDiffSeg reference implementation.

Built from the architecture description in the DualDiffSeg manuscript
(Sections 3.1-3.6). Every component carries an ablation switch so that all
eight configurations of the ablation table are produced by one class rather
than eight forked copies.

Component provenance, for the record:
  - S-FEM and RPE follow DBL-Net (Zhu et al., 2024), adapted from 2D
    ultrasound to 3D MRI.
  - CAFU follows the confidence-aware fusion principle of Li et al. (2022).
  - The diffusion branch is iterative residual refinement under a smoothness
    constraint. It is NOT a denoising diffusion probabilistic model: there is
    no forward noising process, no timestep embedding, and no variational
    objective. Describe it accordingly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def make_norm(kind, ch):
    """
    Normalization layer factory.

    BatchNorm computes statistics over the batch, which is unreliable at the
    batch sizes volumetric segmentation forces (2-4). nnU-Net uses
    InstanceNorm throughout for exactly this reason: it normalizes per-sample
    per-channel and is independent of batch size.

    Measured effect here: the manuscript specifies BatchNorm, and at batch 2
    that is a plausible contributor to the 0.25 Dice shortfall against the
    published nnU-Net reference.
    """
    if kind == "batch":
        return nn.BatchNorm3d(ch)
    if kind == "instance":
        return nn.InstanceNorm3d(ch, affine=True)
    if kind == "group":
        return nn.GroupNorm(min(8, ch), ch)
    raise ValueError(f"unknown norm {kind!r}")

class ConvBlock(nn.Module):
    """Conv3d -> (BN) -> ReLU, optionally strided."""

    def __init__(self, cin, cout, stride=1, norm=True, norm_kind="batch"):
        super().__init__()
        layers = [nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=not norm)]
        if norm:
            layers.append(make_norm(norm_kind, cout))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SemanticBranch(nn.Module):
    """
    CNN-Transformer hybrid encoder (Section 3.2).

    Three convolutional stages at 32/64/128 channels with stride-2
    downsampling, then a Transformer encoder over the flattened bottleneck
    to model long-range dependencies.

    Returns (bottleneck_features, [skip1, skip2]).
    """

    def __init__(self, in_ch=1, chans=(32, 64, 128), n_layers=2, n_heads=4,
                 use_transformer=True, patch=(96, 96, 32), norm_kind="batch"):
        super().__init__()
        c1, c2, c3 = chans
        nk = dict(norm_kind=norm_kind)
        self.stage1 = nn.Sequential(ConvBlock(in_ch, c1, **nk), ConvBlock(c1, c1, **nk))
        self.stage2 = nn.Sequential(ConvBlock(c1, c2, stride=2, **nk), ConvBlock(c2, c2, **nk))
        self.stage3 = nn.Sequential(ConvBlock(c2, c3, stride=2, **nk), ConvBlock(c3, c3, **nk))

        self.use_transformer = use_transformer
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=c3, nhead=n_heads, dim_feedforward=c3 * 4,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

            # Positional embedding is created HERE, not lazily in forward().
            # A parameter created during the first forward pass is missing
            # from optimizer.param_groups (the optimizer is built first), so
            # it receives gradients but is never updated — it stays frozen at
            # random init for the whole run. Sizing it at construction from
            # the patch shape keeps it trainable.
            #
            # Two stride-2 stages, so the bottleneck is patch // 4 per axis.
            n_tokens = 1
            for p in patch:
                n_tokens *= max(1, p // 4)
            self.n_tokens = n_tokens
            self.pos = nn.Parameter(torch.zeros(1, n_tokens, c3))
            nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        if self.use_transformer:
            B, C, D, H, W = s3.shape
            tok = s3.flatten(2).transpose(1, 2)               # B, N, C

            pos = self.pos
            if pos.shape[1] != tok.shape[1]:
                # Patch size differs from the one the model was built for —
                # happens during sliding-window inference on odd edge regions.
                # Interpolate rather than rebuild, so the learned embedding is
                # preserved and nothing is silently reinitialized.
                pos = torch.nn.functional.interpolate(
                    pos.transpose(1, 2), size=tok.shape[1],
                    mode="linear", align_corners=False,
                ).transpose(1, 2)

            tok = self.transformer(tok + pos)
            s3 = tok.transpose(1, 2).reshape(B, C, D, H, W)

        return s3, [s1, s2]


class RefinementBlock(nn.Module):
    """Conv3d -> BN -> ReLU with a residual connection (Section 3.2)."""

    def __init__(self, ch, norm_kind="batch"):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.norm = make_norm(norm_kind, ch)

    def forward(self, x):
        return F.relu(x + self.norm(self.conv(x)))


class DiffusionBranch(nn.Module):
    """
    Diffusion-inspired refinement branch (Section 3.2).

    Two initial convolutions extract low- to mid-level features, then T
    residual refinement stages run at matched resolution. The list of
    intermediate feature maps is returned so that the diffusion-consistency
    loss can penalize the difference between consecutive stages.
    """

    def __init__(self, in_ch=1, chans=(32, 64, 128), n_stages=3, norm_kind="batch"):
        super().__init__()
        c1, c2, c3 = chans
        self.stem = nn.Sequential(
            ConvBlock(in_ch, c1, norm=False), ConvBlock(c1, c1, norm=False)
        )
        self.down1 = ConvBlock(c1, c2, stride=2, norm_kind=norm_kind)
        self.down2 = ConvBlock(c2, c3, stride=2, norm_kind=norm_kind)
        self.stages = nn.ModuleList(
            [RefinementBlock(c3, norm_kind=norm_kind) for _ in range(n_stages)])

    def forward(self, x):
        h = self.down2(self.down1(self.stem(x)))
        traj = [h]
        for stage in self.stages:
            h = stage(h)
            traj.append(h)
        return h, traj


class SFEM(nn.Module):
    """
    Spatial-Frequency Encoding Module (Section 3.3).

    A 3D FFT branch (1x1 conv on the real/imag spectrum) runs in parallel with
    a spatial 3D convolution; the two are concatenated and fused by a learned
    channel attention.
    """

    def __init__(self, ch, norm_kind="batch"):
        super().__init__()
        self.spatial = nn.Conv3d(ch, ch, 3, padding=1)
        self.freq = nn.Conv3d(ch * 2, ch, 1)
        self.fuse = nn.Sequential(
            nn.Conv3d(ch * 2, ch, 1), make_norm(norm_kind, ch), nn.ReLU(inplace=True)
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Conv3d(ch, ch // 4, 1),
            nn.ReLU(inplace=True), nn.Conv3d(ch // 4, ch, 1), nn.Sigmoid()
        )

    def forward(self, x):
        spatial = self.spatial(x)

        spec = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))
        spec = torch.cat([spec.real, spec.imag], dim=1)
        spec = self.freq(spec)
        # back to the spatial grid; rfftn halves the last axis
        spec = F.interpolate(spec, size=x.shape[-3:], mode="trilinear",
                             align_corners=False).to(x.dtype)

        fused = self.fuse(torch.cat([spatial, spec], dim=1))
        return fused * self.attn(fused)


class RPE(nn.Module):
    """Representational Perception Enhancer (Section 3.4)."""

    def __init__(self, ch):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(ch, ch // 4), nn.ReLU(inplace=True),
            nn.Linear(ch // 4, ch), nn.Sigmoid()
        )
        self.sa = nn.Sequential(nn.Conv3d(ch, 1, 7, padding=3), nn.Sigmoid())
        self.res = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1)
        )

    def forward(self, x):
        c = self.ca(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        s = self.sa(x)
        y = x * c * s
        return F.relu(x + self.res(y))


class CAFU(nn.Module):
    """
    Confidence-Aware Fusion Unit (Section 3.5).

    Produces voxel-wise confidence weights over the two branches via softmax,
    then forms the adaptively weighted sum. The confidence map is returned so
    that the entropy regularizer can be applied to it.
    """

    def __init__(self, ch, norm_kind="batch"):
        super().__init__()
        self.conf = nn.Conv3d(ch * 2, 2, 3, padding=1)
        self.proj = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1), make_norm(norm_kind, ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, sem, dif):
        w = torch.softmax(self.conf(torch.cat([sem, dif], dim=1)), dim=1)
        fused = w[:, 0:1] * sem + w[:, 1:2] * dif
        return self.proj(fused), w


class Decoder(nn.Module):
    """Three-level transposed-convolution decoder with skip connections."""

    def __init__(self, chans=(32, 64, 128), n_classes=1, norm_kind="batch"):
        super().__init__()
        c1, c2, c3 = chans
        nk = dict(norm_kind=norm_kind)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(c2 * 2, c2, **nk), ConvBlock(c2, c2, **nk))
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock(c1 * 2, c1, **nk), ConvBlock(c1, c1, **nk))
        self.head = nn.Conv3d(c1, n_classes, 1)

    def forward(self, x, skips):
        s1, s2 = skips
        x = self.up2(x)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, s1], dim=1))
        return self.head(x)


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------

class DualDiffSeg(nn.Module):
    """
    Full DualDiffSeg network with ablation switches.

    Ablation flags map to the rows of the ablation table:

        use_diffusion_branch=False              -> single-branch baseline
        use_cafu=False                          -> concatenation fusion
        use_sfem=False                          -> no S-FEM
        use_rpe=False                           -> no RPE

    Loss-term ablations (L_diff, L_conf, L_MRI) are handled in losses.py by
    zeroing the corresponding lambda; the architecture is unchanged.

    forward() returns a dict:
        logits     — segmentation logits, B x n_classes x D x H x W
        conf       — CAFU confidence map (None if use_cafu=False)
        diff_traj  — list of diffusion-branch stage outputs (None if disabled)
    """

    def __init__(self, in_ch=1, n_classes=1, chans=(32, 64, 128),
                 n_stages=3, n_layers=2, n_heads=4,
                 use_diffusion_branch=True, use_sfem=True,
                 use_rpe=True, use_cafu=True, use_transformer=True,
                 patch=(96, 96, 32), norm_kind="batch"):
        super().__init__()
        c3 = chans[-1]
        self.cfg = dict(
            use_diffusion_branch=use_diffusion_branch, use_sfem=use_sfem,
            use_rpe=use_rpe, use_cafu=use_cafu, use_transformer=use_transformer,
        )

        self.norm_kind = norm_kind
        self.semantic = SemanticBranch(in_ch, chans, n_layers, n_heads,
                                       use_transformer=use_transformer,
                                       patch=patch, norm_kind=norm_kind)
        self.diffusion = (DiffusionBranch(in_ch, chans, n_stages,
                                          norm_kind=norm_kind)
                          if use_diffusion_branch else None)

        if use_sfem:
            self.sfem_sem = SFEM(c3, norm_kind=norm_kind)
            self.sfem_dif = (SFEM(c3, norm_kind=norm_kind)
                             if use_diffusion_branch else None)
        if use_rpe:
            self.rpe_sem = RPE(c3)
            self.rpe_dif = RPE(c3) if use_diffusion_branch else None

        if use_diffusion_branch:
            if use_cafu:
                self.fuse = CAFU(c3, norm_kind=norm_kind)
            else:
                self.fuse = nn.Sequential(
                    nn.Conv3d(c3 * 2, c3, 3, padding=1),
                    make_norm(norm_kind, c3), nn.ReLU(inplace=True)
                )

        self.decoder = Decoder(chans, n_classes, norm_kind=norm_kind)

    def forward(self, x):
        sem, skips = self.semantic(x)
        traj = None

        if self.diffusion is not None:
            dif, traj = self.diffusion(x)
        else:
            dif = None

        if self.cfg["use_sfem"]:
            sem = self.sfem_sem(sem)
            if dif is not None:
                dif = self.sfem_dif(dif)
        if self.cfg["use_rpe"]:
            sem = self.rpe_sem(sem)
            if dif is not None:
                dif = self.rpe_dif(dif)

        conf = None
        if dif is None:
            fused = sem
        elif self.cfg["use_cafu"]:
            fused, conf = self.fuse(sem, dif)
        else:
            fused = self.fuse(torch.cat([sem, dif], dim=1))

        return dict(logits=self.decoder(fused, skips), conf=conf, diff_traj=traj)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# ablation configurations
# ---------------------------------------------------------------------------

ABLATIONS = {
    "single_branch":  dict(use_diffusion_branch=False, use_sfem=True,  use_rpe=True,  use_cafu=True),
    "no_cafu":        dict(use_diffusion_branch=True,  use_sfem=True,  use_rpe=True,  use_cafu=False),
    "no_sfem":        dict(use_diffusion_branch=True,  use_sfem=False, use_rpe=True,  use_cafu=True),
    "no_rpe":         dict(use_diffusion_branch=True,  use_sfem=True,  use_rpe=False, use_cafu=True),
    "no_ldiff":       dict(use_diffusion_branch=True,  use_sfem=True,  use_rpe=True,  use_cafu=True),
    "no_lconf":       dict(use_diffusion_branch=True,  use_sfem=True,  use_rpe=True,  use_cafu=True),
    "no_lmri":        dict(use_diffusion_branch=True,  use_sfem=True,  use_rpe=True,  use_cafu=True),
    "full":           dict(use_diffusion_branch=True,  use_sfem=True,  use_rpe=True,  use_cafu=True),
}

# loss-weight overrides for the three loss-term ablations
LOSS_OVERRIDES = {
    "no_ldiff": dict(lambda_diff=0.0),
    "no_lconf": dict(lambda_conf=0.0),
    "no_lmri":  dict(lambda_mri=0.0),
}


def build(config="full", **kwargs):
    if config not in ABLATIONS:
        raise ValueError(f"unknown config {config!r}; choose from {list(ABLATIONS)}")
    return DualDiffSeg(**{**ABLATIONS[config], **kwargs})
