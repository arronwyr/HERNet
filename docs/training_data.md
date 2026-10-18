# Synthetic training-data preparation

## Assets and environment

Install the optional data dependencies:

```bash
python -m pip install -r requirements-data.txt
```

Obtain Anguli synthetic fingerprints, paired binary clean targets, background textures, and an NBIS 5.0.0 MINDTCT executable. They are external assets and are not redistributed here. The generation code uses MINDTCT to derive auxiliary supervision from the clean target, not from enhanced test images. The tested MINDTCT build supports 8-bit grayscale PNG input.

Organize clean source assets as follows; relative paths under `image/` and `label/` must match:

```text
source/
  image/folder/sample.png
  label/folder/sample.png
textures/
  background_001.png
  background_002.png
```

Use 512x512 source pairs and textures for the manuscript setting. Textures of a different size are resized to the source size as an input-compatibility fallback. Source fingerprints and background textures should be obtained and used under their providers' terms. The manuscript uses road-crack, steel-surface-defect, magnetic-tile, and solar-cell textures. Data order and asset versions influence the generated sample sequence even with a fixed random seed.

## Generate images

```bash
python data_preparation/generate.py --ims_dir /path/to/source --txs_dir /path/to/textures --out_dir /path/to/generated --out_num 40000 --random_seed 42 --q_th 0.25 --mindtct /path/to/mindtct
```

On Windows, use the corresponding path to `mindtct.exe`. Paths containing spaces should be quoted. Start with `--out_num 2` to check the environment before generating the complete set. A timestamped result directory is created inside `--out_dir`. `--start_n` offsets the repeat index used in output filenames.

Generation follows the manuscript's non-v2 procedure: paired flipping/rotation/scaling; scars and morphological wet/dry artifacts; global and additional local ridge degradation; shared plastic distortion and partial-impression masking; and blending of the corrupted input with a background texture. The target shares geometry with the input but not its appearance corruption.

### Ridge-degradation parameters

| Parameter | Setting |
|---|---|
| Global filtering choice | Directional diffusion 0.5; Gaussian blur 0.5 |
| Diffusion coefficients | c1=0.9, c2=0.1 |
| Diffusion update step | s=0.5 |
| Global diffusion iterations | 5 |
| Additional local diffusion iterations | Uniform integer 3-8 |
| Diffusion direction | Clean ridge field 0.4; field plus a sampled rotation 0.2; constant sampled orientation 0.4 |
| Sampled angular offset | Uniform [-pi/2, pi/2) |
| Global Gaussian filtering | Three applications of a 5x5 Gaussian filter |
| Additional local Gaussian filtering | Uniform integer 5-11 applications |
| Local diffusion mask noise frequency | Uniform integer 32-63 |
| Local Gaussian mask noise frequency | Uniform integer 16-47 |
| Mask threshold multiplier | Uniform [0.98, 1.02) times the noise-map mean |
| Input/background blending weight | Uniform [0.35, 0.65) |
| MINDTCT quality threshold for supervision | 0.25 |
| Minutiae exclusion near mask boundary | 16 pixels |
| Minutiae heatmap kernel | Gaussian 35x35, sigma=8 |

Global and local diffusion use the same selected direction field. Local Gaussian filtering operates on a copy of the globally blurred image and replaces pixels only within the selected local mask. The generated parameter records are saved per sample in `value/`.

Outputs include `image/`, `label/`, `mask/`, `orientation/`, and `minutia/`. Orientation arrays contain the ridge angle in radians. Minutiae arrays contain the Gaussian-filtered target map in its original numerical scale. `information/` contains MINDTCT diagnostic files; only the five training fields are packed into HDF5.

## Convert to HDF5

```bash
python data_preparation/to_hdf5.py --input /path/to/generated/generated_TIMESTAMP --output /path/to/train_ilsom.hdf5
```

The default image size is 512. The converter checks companion files, field dimensions, and finite values, and refuses to overwrite an existing archive.

| Key | Shape | Type | Meaning |
|---|---|---|---|
| img | N x 1 x 512 x 512 | uint8 | Degraded input |
| lbl | N x 1 x 512 x 512 | uint8 | Aligned clean target |
| msk | N x 1 x 512 x 512 | uint8 | Foreground mask, 0/255 |
| ort | N x 2 x 512 x 512 | float16 | cos(2 theta), sin(2 theta), zero outside foreground |
| mnt | N x 1 x 512 x 512 | float16 | Auxiliary minutiae-map target |
| filenames | N | UTF-8 string | Relative input filenames |

The `image_number` attribute stores N. Image, target, and mask arrays are divided by 255 when used as floating-point training inputs; orientation and minutiae arrays already contain floating-point quantities. The conversion does not normalize the minutiae map to unit peak.

The manuscript trains for 10 epochs with batch size 8, learning rate 1e-4 halved every 3 epochs, and auxiliary loss weights of 1. Reconstruction is foreground-weighted for the first two epochs and covers the whole image thereafter. Both auxiliary losses remain foreground-weighted. The training-only heads can be instantiated with `HERNet(auxiliary=True)`; the inference checkpoint cannot supply their omitted tensors.
