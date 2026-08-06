# ToPo-OPD

Official project repository for **Topology-Calibrated Prototype Distillation for Few-shot Point Cloud Segmentation**.

> This first release provides the core model components. Training, dataset, and evaluation scripts will be added after the experiment pipeline is finalized.

## Overview

Few-shot point cloud semantic segmentation (FS-PCS) aims to segment novel semantic categories from only a few annotated support scenes. Existing prototype-based approaches are often affected by two coupled problems:

- Support prototypes produced by masked average pooling can discard local topology and become biased under scarce supervision.
- Coarse query representations are not explicitly aligned with the support prototype space, which can lead to fragmented predictions.

**ToPo-OPD** addresses both problems through topology-guided prototype calibration and on-policy prototype distillation.

## Method

### Topology-guided Prototype Calibration (ToPo)

ToPo expands the initial support prototype into multiple candidates. It then estimates topology-aware weights from masked support features and aggregates the candidates into a calibrated prototype.

This design preserves structural information from the support point cloud and reduces prototype bias caused by limited support samples and intra-class geometric variation.

### On-policy Prototype Distillation (OPD)

OPD constructs a query-side prototype from the model's current coarse query activations. During training, the query prototype is aligned with the calibrated support prototype in a shared embedding space.

OPD regularizes the query states that the model actually visits during optimization. It is used only during training and introduces no additional inference-time branch.

## Backbones and Planned Experiments

The current manuscript uses DGCNN following the established few-shot point cloud segmentation pipeline. The code also exposes a Stratified Transformer adapter so that the same ToPo and OPD modules can be evaluated with a stronger hierarchical point encoder without coupling this repository to custom CUDA operators.

Future experiments will study additional backbone families, topology neighborhood settings, candidate counts, distillation temperatures, and cross-dataset generalization. Verified configurations and results will be published progressively.

## Highlights

- Jointly addresses support-side prototype bias and query-side misalignment.
- Uses topology-aware candidate aggregation to improve prototype reliability.
- Performs prototype-level, on-policy distillation between support and query representations.
- Evaluated under standard 2-way/3-way and 1-shot/5-shot settings.
- Experiments cover the S3DIS and ScanNet indoor point cloud benchmarks.
- The current manuscript reports the best mean IoU in seven of eight evaluated settings.

## Repository Status

Available now:

- Core DGCNN backbone
- Stratified Transformer adapter
- Topology-guided prototype calibration
- On-policy prototype distillation
- End-to-end episodic model composition

To be added progressively:

- Training and evaluation code
- Dataset preparation instructions
- Experiment configurations
- Pretrained checkpoints
- Reproduction commands
- Detailed quantitative and qualitative results

## Citation

Citation information will be added after the paper metadata is finalized.

## Acknowledgement

We thank the authors of DGCNN and Stratified Transformer for their excellent work.
