# HERNet

Implementation and supplementary results for **Hierarchical Spatial Routing of Heterogeneous Experts for Latent Fingerprint Enhancement**.

HERNet combines a fine-scale convolutional expert (FCE), a long-range Transformer expert (LTE), and a coarse-scale convolutional expert (CCE) through two stages of spatial soft routing. Orientation-field and minutiae-map heads provide auxiliary supervision during training; the released enhancement checkpoint does not contain these heads.

## Contents

```text
hernet/                  Model definitions and supporting neural-network blocks
enhance.py               Directory-based image enhancement
weights/                 Inference checkpoint and SHA-256/verification manifest
data_preparation/        Synthetic training-pair generation and HDF5 conversion
evaluation/              CMC computation and plotting tools
supplementary/cmc/       Curves, numeric data, and manuscript consistency audit
docs/                    Data, matching, and baseline implementation protocols
tests/                   Lightweight model tests
```

## Installation

Python 3.10 or newer is recommended. Install a PyTorch build appropriate for your CPU/CUDA environment, then install the inference dependencies:

```bash
python -m pip install -r requirements.txt
```

The release was checked with Python 3.12 and PyTorch 2.8.0. TensorFlow, NBIS, VeriFinger, and the baseline repositories are **not** required to run HERNet enhancement.

## Enhance a folder

Place `hernet_inference.pth` in `weights/`. The local distribution contains this file; when distributed separately, obtain the same file and verify its SHA-256 against `weights/manifest.json`.

```bash
python enhance.py --input /path/to/latent_images --output /path/to/enhanced_images --device cuda:0
```

For CPU inference:

```bash
python enhance.py --input /path/to/latent_images --output /path/to/enhanced_images --device cpu
```

Use `--recursive` for nested input folders or `--weights /path/to/hernet_inference.pth` for a different checkpoint location. Output is grayscale PNG with the original spatial dimensions and relative directory structure. Existing output files are protected unless `--overwrite` is specified. The output directory must be outside the input directory.

Inputs are 8-bit grayscale or RGB images. RGB channels are averaged, matching the original inference preprocessing. No automatic polarity inversion, resizing, ROI segmentation, or binarization is applied. Use the same ridge polarity and approximately 500-ppi scale as the evaluation images. High-bit-depth TIFFs must first be converted to properly calibrated 8-bit images.

Inference uses replicate padding to dimensions divisible by 32, followed by an additional 32-pixel border on each side; output is center-cropped back to the original dimensions. Per-image standardization is performed **after padding**. The inference parameter count is 26,103,164. Removing the auxiliary heads leaves enhancement output unchanged; verification records are included in the weight manifest.

## Training-data preparation

See [the training-data guide](docs/training_data.md) for required assets, generation parameters, NBIS requirements, and the HDF5 layout. The tools prepare only training data; benchmark image databases and third-party model files are obtained from their respective providers.

## Supplementary recognition results

The [CMC overview](supplementary/cmc/CMC_all_datasets_matchers.png) contains three datasets and four matching pipelines. Individual PNG/SVG figures and full-rank numerical curves are in [supplementary/cmc](supplementary/cmc). All 92 method/dataset/pipeline curves reproduce the manuscript's rounded Rank-1 and Rank-20 values.

- [Matching and evaluation protocol](docs/matching_protocol.md)
- [Enhancement baseline implementation details](docs/baselines.md)
- [CMC data format and reproduction](docs/cmc_results.md)

## Tests

```bash
python -m unittest discover -s tests -v
```

This distribution contains the model definition, inference entry point, and training-data tools. It does not bundle benchmark images, proprietary matcher software, a training driver, or third-party baseline weights.

## Research use

Enhanced ridge appearance is not a guarantee of authentic structure recovery. Outputs should not be treated as independent forensic evidence or replace examination of the original impression. Recognition and annotation-coverage evaluations assess particular downstream properties, not complete forensic validity.

See [third-party acknowledgments](THIRD_PARTY_NOTICES.md). Citation metadata is provided in `CITATION.cff`; the associated manuscript is not represented here as an accepted publication.
