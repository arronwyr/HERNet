# Recognition protocols and matcher settings

## Datasets and relevance

| Dataset | Probes | Gallery images | Relevant gallery entries |
|---|---:|---:|---|
| MOLF | 4,400 DB4 latents | 4,000 DB3A prints | Same subject |
| MUST | 13,577 latents from ten test subjects | 1,080 prints | Same subject and finger |
| NIST SD302 | 1,005 SD302H latents from 119 subjects | 2,000 SD302B V500 rolled prints | Same subject and finger |

MOLF uses subject-level retrieval because of potential finger-level correspondence inconsistencies; its scores should not be interpreted as finger-level identification. The MUST gallery contains 100 db36 genuine-reference prints and 980 db39/MOLF distractor prints. MUST subjects 10378 and 53759 are held out for validation/checkpoint selection and are distinct from the ten test subjects. SD302 probes follow the author-provided SFP evaluation outputs and current correspondence, retaining 1,005 of the historical 1,019 entries.

Within a matching pipeline, the probe/gallery identity lists are the same for all compared enhancements. Only latent probes are enhanced. Gallery images remain the original reference impressions, with any preprocessing intrinsic to the matcher retained. Matching scores are not comparable numerically across pipelines.

## 1. NBIS

- Version: NBIS 5.0.0, MINDTCT extraction followed by Bozorth3 matching.
- Extraction input: 8-bit grayscale PNG with 500-ppi metadata; no command-line extraction threshold override.
- Extractor call: `mindtct input.png output_prefix`; use the emitted `.xyt` file.
- Matching call: `bozorth3 probe.xyt gallery.xyt`, or the equivalent gallery-list interface `bozorth3 -p probe.xyt -G gallery_list.txt`.
- XYT fields: x, y in the NBIS coordinate convention, integer-degree direction, and quality. No manual ROI is supplied to this pipeline.
- All probes are compared with the same gallery, producing a probe-by-gallery matrix.

**Historical score storage:** the NBIS score matrices used for the manuscript cap Bozorth3 integer scores at 127 and store them as int8. The distributed CMC data reproduce those matrices without changing their scores. This is a legacy storage choice, not a recommended numerical representation for new matching experiments; new implementations should use an adequately sized integer type and document any change in protocol.

## 2. LEADER + Bozorth3

- Extractor: pretrained `pyfing.Leader` with `LEADER.weights.h5`.
- Image scale: 500 dpi; quality threshold: 0.6.
- Other extractor settings retain the local implementation defaults: input-size multiple 32 and minutia type threshold 0.5.
- Detection uses the original per-image inference behavior; equal-size batching preserves the per-image padding geometry.
- Coordinates are converted from image-style top-left origin to NBIS XYT: `y_NBIS = H - 1 - int(y)`.
- Direction is converted by `(round(degrees(direction)) + 180) % 360`; quality is rounded to 0-100.
- Original gallery and enhanced probe images are extracted separately, then matched with Bozorth3 as above. The historical score-storage cap also applies to this pipeline.

LEADER is used as a learned minutiae extractor, not as an enhancement algorithm. Comparison with NBIS changes the extractor while retaining Bozorth3 as the matcher.

## 3. VeriFinger

- SDK: VeriFinger 2025.2; proprietary software and a valid provider-issued license are required separately.
- Extraction: grayscale images with 500-ppi metadata, using the SDK's `FingerEngine.extract_finger`; exported templates are matched using `NBiometricEngine.verify_offline`.
- No extraction-quality, matching-speed, or matching-threshold override is explicitly set by the wrapper; the SDK defaults apply.
- The recorded score matrix uses int16. An extraction failure produces an empty template file; a verification exception is assigned score 0 by the wrapper.
- The original gallery templates are shared across enhancement methods.

SDK binaries, activation information, licenses, and gallery templates are not distributed with this repository.

## 4. FLARE / FDD

- Pose model: pretrained VotingPose, `VotingPose.pth`, configured for 500 ppi and a 512x512 middle shape.
- Descriptor: pretrained FDD, `desc_model.pth.tar`, `ndim_feat=6`, `input_norm=False`, target shape 256x256.
- **Fixed-pose comparison:** pose is estimated on each original latent and reused by all its enhanced versions. Original gallery poses and features are shared. Pose is not independently re-estimated on every enhanced image.
- Descriptor input: affine alignment with spatial scale 0.5 and center [128,128], bilinear image sampling to 256x256, intensities normalized by `(I-127.5)/127.5`.
- FingerNet/Autoenc stride cropping preserves the top-left origin; original-image pose coordinates are retained without dimension-ratio rescaling.
- Scoring: mask-weighted cosine similarity, with the product of the masked feature norms clamped below at 1e-3; output scores are float32. Mask weights are expanded to match feature channels.

For features f,g and expanded masks m,n, the score is:

```text
sum(m*n*f*g) / max(sqrt(sum(m*n*f*f))*sqrt(sum(m*n*g*g)), 1e-3)
```

UNetEnh's enhancement checkpoint is distinct from the FLARE pose/descriptor checkpoints. The evaluated FingerGAN input is its direct enhancement output, not the separate maximum-filtered variant.

## Identification and verification

CMC uses gallery-image candidates and the first relevant result; see [CMC reproduction](cmc_results.md). Gallery indices tied in score follow the original NumPy argsort ordering. MOLF reports subject-level identification only.

MUST and SD302 verification uses every same-finger probe/gallery pair as genuine and all other pairs as impostor, without impostor subsampling. MUST has 13,577 genuine and 14,649,583 impostor pairs; SD302 has 1,005 genuine and 2,008,995 impostor pairs.

EER is obtained from distinct observed score thresholds with interpolation at the FAR/FRR crossing. TAR at FAR=0.001 uses the most permissive available threshold satisfying FAR <= 0.001; TAR is not interpolated to an unattainable operating point. Stored zero-score/failure entries remain in the evaluation denominators. A failure policy should be recorded separately for each matcher rather than assuming identical behavior across all four pipelines.

## Sources

- [NIST Biometric Image Software](https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis)
- [LEADER paper](https://arxiv.org/abs/2602.15493)
- [VeriFinger SDK](https://www.neurotechnology.com/verifinger.html)
- [FLARE repository](https://github.com/Yu-Yy/FLARE)

This document records the implemented evaluation protocol. Third-party internal defaults not explicitly exposed by a wrapper are not presented as independently tuned hyperparameters. Source/checkpoint hashes for locally archived components are listed in `baseline_components.json`.
