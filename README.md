# Dual-Branch Diffusion-Aware Segmentation of Breast Tumours in DCE-MRI

Code and results accompanying the manuscript *"Dual-Branch Diffusion-Aware Segmentation of Breast Tumours in DCE-MRI: Localisation, Partition Composition, and Post-Processing Dominate Architectural Choice."*

We evaluate a dual-branch architecture — semantic feature extraction, relative positional encoding, and confidence-aware fusion — on the [MAMA-MIA](https://github.com/LidiaGarrucho/MAMA-MIA) benchmark under the official patient-level partition, with full-volume evaluation.

**The architecture does not outperform the benchmark's nnU-Net baseline**, and an eight-configuration ablation finds no component contributing a practically meaningful improvement. The analysis of *where* the performance is lost is the substance of the work.

## Findings

| | |
|---|---|
| **Localisation, not segmentation** | Validation Dice on lesion-centred patches reaches 0.735–0.788 across all eight configurations; full-volume test Dice for the same checkpoints is 0.491–0.539. The gap is −0.24 to −0.27 for every configuration (spread 0.024), so it is attributable to no architectural element. |
| **Partition imbalance** | The official split is compositionally imbalanced by source collection (χ² = 96.89, p = 1.1 × 10⁻¹⁸). Train is 71.1% ISPY2 against 42.8% in test. Reweighting to the training composition improves Dice by 0.047–0.068 — roughly a quarter of the deficit. |
| **Post-processing > architecture** | A confidence-conditional component selection rule, selected on validation and confirmed on test, improves Dice by 0.0380 with no retraining — more than the total spread of the architectural ablation. ~0.12 Dice remains available to an oracle selector. |

## Repository layout

```
train.py                  training, all eight ablation configurations
models.py                 architecture (SFEM, RPE branch, CAFU)
losses.py                 composite objective
preprocess.py             resampling and normalisation
prepare_data.py           manifest construction
evaluate.py               full-volume sliding-window evaluation
tune_threshold.py         validation threshold sweep
component_analysis.py     connected-component diagnostics
selection_rules.py        selection-rule comparison
final_analysis.py         oracle selection, rule sweep, corrected metrics
rescore.py                CPU-only rescoring from saved probability maps
check_orientation.py      NIfTI header audit for the axis-order defect
manifests/                official partition (1,020 / 180 / 306)
results/                  per-case metrics, ablation tables, significance tests
```

## Reproducing

```bash
# 1. obtain MAMA-MIA (Synapse syn60868042 or the Health-RI XNAT mirror)
# 2. preprocess
python preprocess.py --data /path/to/MAMA-MIA --out data/preprocessed
# 3. train (repeat with --config for each of the eight configurations)
python train.py --manifests data/preprocessed/manifests --runs runs --config full
# 4. evaluate at the tuned operating point
python evaluate.py --manifests data/preprocessed/manifests --runs runs \
                   --out results --threshold 0.98
# 5. selection-rule analysis
python final_analysis.py --manifests data/preprocessed/manifests --runs runs \
                         --audit orientation_audit.json --out results_final --probs probs
python rescore.py --probs probs --audit orientation_audit.json --out results_rescored
```

Trained on a single NVIDIA A10G (AWS g5.xlarge), 200 epochs, ~2 hours per configuration.

## Configuration

Single DCE phase (first post-contrast), 96 × 96 × 32 patches, 1.5 × 1.5 × 2.0 mm spacing, per-case z-score normalisation. Adam, lr 1e-4, weight decay 1e-5, batch 2, mixed precision, seed 42 for all configurations.

Composite loss: α = 0.5, λ_seg = 1.0, λ_diff = 0.1, λ_conf = 0.05, λ_mri = 0.01, with confidence warmup 20 epochs, ramp 30, floor 0.05.

Operating threshold 0.98, selected on validation.

## A note on preprocessing

We use 1.5 × 1.5 × 2.0 mm spacing and single-phase normalisation. The MAMA-MIA authors recommend 1.0 mm isotropic and all-phase normalisation. Results are conditioned on this deviation; see §3.7 of the manuscript.

`check_orientation.py` audits an axis-order defect in resampling that affects 15.6% of cases. It is corrected in the released code, and we show it does not explain the failure mode (OR 1.06, p = 0.881), but it will arise in any pipeline that reorients before resampling.

## Data

The MAMA-MIA dataset is available via [Synapse](https://www.synapse.org/#!Synapse:syn60868042) (CC BY-NC 4.0) and the Health-RI XNAT platform. Constituent TCIA collections (ISPY1, ISPY2, Breast-MRI-NACT-Pilot, Duke-Breast-Cancer-MRI) carry their own licences and must be cited separately.

## Citation

```bibtex
@article{thirupathi2026dualbranch,
  title   = {Dual-Branch Diffusion-Aware Segmentation of Breast Tumours in DCE-MRI:
             Localisation, Partition Composition, and Post-Processing Dominate
             Architectural Choice},
  author  = {Thirupathi, Perugu and Devi, Bishnulatpam Pushpa and Kumar, T. Kishore},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

Please also cite the MAMA-MIA dataset (Garrucho et al., *Sci. Data* **12**, 453, 2025).

## Licence

Code released under the MIT Licence. Dataset licences are as stated above.
