# CMC supplementary results

There are 12 dataset/pipeline panels and 92 individual curves, including Raw and HERNet. The panels cover MOLF, MUST, and NIST SD302 with NBIS, VeriFinger, LEADER+Bozorth3, and FLARE. SFP author-provided outputs are available for the evaluated MOLF/SD302 subsets; its MUST result is unavailable under this comparison protocol.

## Files

- `CMC_all_datasets_matchers.png` and `.svg`: 4x3 overview.
- `<PIPELINE>_<DATASET>.png` and `.svg`: individual plots, Rank 1-50.
- `<PIPELINE>_<DATASET>.csv`: one column per method and one row per rank, from 1 to the full gallery size. Values are percentages, not fractions.
- `rank1_rank20_audit.csv`: unrounded Rank-1/Rank-20 rates for all 92 curves. They agree with the manuscript values rounded to two decimal places.
- `manifest.json`: matrix hashes, dimensions/dtypes, ordered-list hashes, NumPy version, and consistency checks. Local filesystem paths are not needed to use these files.

These are score-matrix-derived identification curves, not interpolations between the manuscript's Rank-1 and Rank-20 points. Only the figure display range is restricted to 50. No smoothing or extrapolation is used.

## Compute a CMC from another score matrix

```bash
python evaluation/cmc.py --scores scores.npy --probes probe_files.txt --gallery gallery_files.txt --dataset must --output cmc.csv
```

The matrix must be probe-by-gallery, with larger values indicating a better match. Text files contain one filename/path per line in the same row/column order. Filename identities follow the dataset conventions described in [the matching protocol](matching_protocol.md).

Ranking uses `np.argsort(scores, axis=1)[:, ::-1]` on the original matrix dtype. Success at Rank-k means that at least one relevant gallery **image** appears within the top k candidates; gallery images are not collapsed into subject identities. Tied-score behavior follows the recorded NumPy implementation and gallery ordering, rather than optimistic rank computed solely by counting strictly greater scores. This distinction matters for integer matcher scores.

The distributed numeric curves are the reference for the reported results. Different sorting implementations can give different ordering within ties even when the score matrix is unchanged.

Figures can be regenerated from the distributed CSV files:

```bash
python -m pip install -r requirements-evaluation.txt
python evaluation/plot_cmc.py --data supplementary/cmc --output outputs/cmc --max-rank 50
```
