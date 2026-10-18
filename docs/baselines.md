# Enhancement baseline implementations

The comparison uses publicly released pretrained implementations or author-provided enhanced images, without retraining on HERNet's synthetic training pairs. Baselines operate through their own intended preprocessing/postprocessing; they are not converted into a common reconstructed architecture. This repository does not redistribute third-party models or benchmark images.

## FingerNet (2017)

Source: [FingerNet](https://github.com/felixTY/FingerNet), released `Model.model`.

The enhancement export uses the selected Gabor **real response** with the predicted segmentation mask, rather than the phase-angle output. Segmentation is rounded, opened with a 5x5 rectangular structuring element, and nearest-neighbor upsampled by eight. The masked response is byte-scaled with the legacy SciPy-compatible min/max mapping. Inputs are grayscale floats in [0,1]. Bottom/right pixels beyond a multiple of eight are cropped; the top-left coordinate origin is unchanged. Equal-size batching does not add dataset-wide image padding in this export path.

## Autoenc (2020)

This is the enhancement autoencoder component of Cao et al., *End-to-End Latent Fingerprint Search*, not the full LatentAFIS recognition system. The archived component is `Enhancement_AEC_128_depth_4_STFT`, using `model-131000` and its TensorFlow graph.

The latent is treated as 500 ppi, cropped at the bottom/right to a multiple of 16, preprocessed with `FastCartoonTexture(sigma=2.5)` and STFT, then passed to `run_whole_image`. The evaluated output is `_AEC_img`, without a manually supplied segmentation mask. The Autoenc label is used consistently in figures; older component paths may contain the name LatentAFIS.

## FingerGAN (2023)

Source: [official LatentEnhancement implementation](https://github.com/HubYZ/LatentEnhancement), using the authors' pretrained weights.

The upstream enhancement implementation uses 192x192 sliding windows. Its published local CLI defaults are step 10 and enhancement batch size 48. These are upstream defaults, not independently verified overrides for every archived run. Images smaller than 192 in either dimension are replicate-padded to a minimum of 192 and center-cropped back after enhancement.

The comparison uses the direct saved enhancement images. A separately generated 2x2 maximum-filtered variant is not the input to the distributed CMC curves. The original weight file is not redistributed, and its binary SHA-256 was not preserved in the local component inventory; no unrelated model is substituted for it.

## SFP (2024)

Source: [Spectral Filter Predictor for Progressive Latent Fingerprint Restoration](https://github.com/skconan/SFP-Progressive-Feedback-Latent-Fingerprint-Restoration).

The comparison uses the enhanced MOLF/SD302 images supplied by the authors, with the manual ROI information of that evaluation setting. The corresponding author-provided MUST output is unavailable in this protocol. These images are used as supplied rather than re-running or retraining SFP under guessed parameters. Provider-side filtering and feedback parameters are therefore those of the supplied benchmark outputs, not new settings selected by HERNet.

For SD302, historical filenames are mapped to the current dataset correspondence and the common 1,005-image subset. Manual ROI access distinguishes SFP from fully automatic baselines and should be considered when interpreting comparisons.

## SNFEN (2026)

Implementation: pretrained `pyfing` components SUFS -> SNFOE -> SNFFE -> SNFEN, at 500 dpi. No manual ROI is supplied.

```python
mask = sufs.run(image)
orientations, _ = snfoe.run(image, mask, 500)
ridge_periods = snffe.run(image, mask, orientations, 500)
enhanced = snfen.run(image, mask, orientations, ridge_periods, 500)
enhanced = cv2.bitwise_and(enhanced, mask)
```

SUFS uses its pretrained automatic segmentation at 500 dpi, with network input-size multiple 64, segmentation threshold 0.5, and border 33. SNFOE, SNFFE, and SNFEN retain their pretrained/default parameter settings (500-dpi network input and size multiple 32). Model-component hashes are in `baseline_components.json`.

Source paper: [Unleashing the Power of Simplicity](https://arxiv.org/abs/2603.19004).

## UNetEnh (2026)

Implementation: the FLARE enhancement module, using `unetenh.pth` and `pre_enh=True`. The enhancement model is not the FLARE descriptor or VotingPose model.

Inputs are read as grayscale. Width and height are each rounded up to a multiple of 16, and images are resized to that shape with OpenCV's default linear interpolation. Intensities are normalized to [-1,1] by `(I-127.5)/127.5`. Predictions are clamped to [0,1], converted to uint8, then resized back to the original dimensions. Samples with the same inference size can be batched. This resize-based preprocessing is distinct from HERNet's replicate padding.

Source: [FLARE](https://github.com/Yu-Yy/FLARE) and the authors' associated enhancement module.

## Geometry and practical reproducibility

FingerNet and Autoenc may return slightly smaller images because of stride cropping. Such images retain the original top-left origin and are not geometrically rescaled to compensate. For point-coordinate evaluation, NBIS y coordinates must be interpreted using the **actual enhanced image height** before comparison with the original annotation frame. White padding used solely to compose illustrative panels is not matcher input.

Pretrained weight hashes identify available local model components; they do not by themselves prove historical CLI arguments or complete environment equivalence. This release distinguishes implemented settings, upstream defaults, and author-supplied outputs rather than inventing unrecorded run parameters. Different framework versions and legacy dependency stacks can affect exact reproduction.
