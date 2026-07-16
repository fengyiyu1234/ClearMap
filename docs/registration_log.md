# Registration experiment log

Dated, append-only. One entry per registration-tuning session. Format:

```
## YYYY-MM-DD
**Problem:** what wasn't working
**Analysis:** what the logs/data actually showed
**Changes:** what was changed, in which files
**Result:** (filled in after running)
**Next:** what to try/check next
```

---

## 2026-07-16

**Problem:** P5 mouse samples (small, shape very different from the DeMBA P5
atlas) register poorly. Sample `s12t` was the working example
(`/data/hdd12tb-1/fengyi/COMBINe/clearmap/TSC/s12t`). A first round of bspline
tuning (`tsc_ver1/s12t` -> `TSC/s12t`: `StandardGradientDescent` ->
`AdaptiveStochasticGradientDescent` + `TransformBendingEnergyPenalty`)
improved things only a little, visually — less than expected.

**Analysis:** Audited `elastix.log` / `IterationInfo.*.txt` for both runs.

- The log's `Final metric value` (`-0.164421` old vs `-0.731294` new) looked
  like a big win, but `ShowExactMetricValue` was `false`, so that number is
  just the *last* stochastically-sampled iteration (`ImageSampler
  RandomCoordinate`, `NewSamplesEveryIteration true`) — not a real quality
  measure. Per-iteration metric swings wildly (e.g. between `-0.66` and
  `~0.0` within a few iterations in both runs).
- Averaging each resolution's first vs. last 20% of iterations (a noise-
  robust proxy) shows both runs actually converge to nearly the same MI cost
  (~`-0.18` to `-0.27`) — real improvement was much smaller than the log
  suggested, consistent with what was seen visually.
- `Metric1Weight 1.0` (the bending-energy regularizer added in the first
  tuning round) was numerically inert: logged `Metric1` (bending energy)
  values are ~`1e-4`, vs `Metric0` (MI) ~`0.1`-`0.3` — roughly a 1000-3000x
  gap. At weight 1.0 the "regularization that was supposed to stop the field
  folding" contributed essentially nothing to the optimized total.
- `UseRandomSampleRegion "true"` + ASGD makes elastix silently disable
  `UseAdaptiveStepSizes` (explicit warning in the log) — undercutting the
  point of switching to ASGD in the first place.
- No `-fMask`/`-mMask` — Mattes MI was computed over the whole image
  including background, which matters more when fixed/moving shapes differ
  this much.
- Affine's `TransformParameters.0.txt` shows a diagonal of
  `[1.47, 0.92, 1.25]` plus ~0.04-0.1 shear terms — real, large, anisotropic
  shape mismatch, not something a small param tweak fixes on its own.
  `AutomaticTransformInitializationMethod` defaults to `GeometricalCenter`
  (bounding-box center, not intensity-weighted), which is sensitive to
  asymmetric empty z-slices in the raw stack biasing the initial translation
  guess.

**Changes:**
- `ClearMap/Resources/Alignment/align_bspline.txt`: `Metric1Weight` `1.0` ->
  `1000` (matches the actual MI/bending-energy magnitude gap found above);
  `UseRandomSampleRegion` `true` -> `false` (stop silently disabling ASGD's
  adaptive step size), removed now-unused `SampleRegionSize`;
  `ShowExactMetricValue` `false` -> `true` (future logs report a real
  full-image metric, not a noisy sample).
- New `ClearMap/Alignment/BrainMask.py`: Otsu threshold + morphology
  (closing, hole-fill, largest-connected-component, small dilation) tissue
  mask generator. Also runnable standalone
  (`python -m ClearMap.Alignment.BrainMask <in.tif> <out.tif>`) and prints a
  suggested `crop_for_registration` bounding box.
- `ClearMap/Alignment/Elastix.py`: `align()` now accepts `fixed_mask` /
  `moving_mask`, passed through as `-fMask` / `-mMask`.
- `cellMap.py`'s `align_to_reference()`: generates a fixed-image brain mask
  (on the exact post-crop image passed as `-f`) and passes it as `fixed_mask`.
  Atlas (moving) mask skipped for now — the atlas is already trimmed/clean,
  the sample is where the background/shape problem actually is.

**Result:** see "Update" below — first real re-run hit two more bugs, now
fixed; registration itself hasn't completed successfully yet.

**Next:**
1. Run `python -m ClearMap.Alignment.BrainMask resampled.tif resampled_mask.tif`
   on `s12t`, eyeball the mask in ImageJ/Fiji, and check the printed
   suggested crop bounds.
2. If background is large/asymmetric in z (or x/y), fill in
   `config.yaml`'s `registration.crop_for_registration` accordingly — this
   should help affine specifically because of the `GeometricalCenter` init
   sensitivity noted above.
3. Re-run `cellMap.py` on `s12t`.
4. Compare quality using the trustworthy `ShowExactMetricValue=true` output
   this time (no need for the first/last-20%-average workaround), plus a
   visual check.
5. If quality is still clearly insufficient after this — i.e. the shape gap
   looks bigger than what a B-spline FFD (locally-supported, smooth,
   limited control-point resolution) can realistically capture — that's the
   signal to move to ANTs SyN (diffeomorphic, handles larger nonlinear
   deformation) rather than keep tuning elastix parameters.

### Update (same day, later)

Repo root moved from `/home/fyu7/My_project/COMBINe/ClearMap` to
`/home/fyu7/My_project/ClearMap` mid-session (stale `COMBINe` path still
referenced in a few places, e.g. `.claude/settings.local.json` — harmless,
just old).

Additional fixes made while actually trying to run this on `s12t`:

- **`cellMap.py` had no way to pick a config file** — `load_config()` was
  hardcoded to a now-nonexistent path. Added `--config` (argparse, mirrors
  `stats_group_compare.py`'s convention). Usage:
  `python cellMap.py --config config_12t.yaml`.
- **Log file now auto-saved to `<data_dir>/log.txt`** (`cellMap.py`, a `_Tee`
  class mirrors stdout/stderr to both the terminal and that file, appending
  per run) — no more manually typing the full data_dir path into a shell
  redirect.
  - Bug (fixed same session): `_Tee` needs `fileno()` or
    `subprocess.Popen(stdout=sys.stdout, ...)` inside `Elastix.align()`
    crashes with `AttributeError: '_Tee' object has no attribute 'fileno'`.
    Fixed by delegating `fileno()` to the real underlying stream — elastix's
    subprocess output isn't duplicated into `log.txt` because of this (it
    bypasses the Python-level Tee entirely), but that's fine since elastix
    already writes its own full log to `<result_directory>/elastix.log`.
- **`crop_for_registration` is in *resampled*-image voxels (20µm), not
  `registration.tif`'s (raw stitched, 2.6×2.6×32µm) voxels.** User had been
  reading crop bounds off `registration.tif` directly. Conversion:
  `resampled_index = raw_index * (stitched_resolution_axis /
  resampled_resolution_axis)` — for this config, ×0.13 in x/y, ×1.6 in z.
  Also: a single-element crop list like `z: [20]` crashes
  (`ValueError: not enough values to unpack`) — needs two elements, e.g.
  `z: [20, null]` (the `null` end works fine, Python slice semantics treat it
  as "to the end").
- **Investigated a suspected intensity/normalization bug**: user saw
  `resampled.tif` looking like a much narrower value range than
  `registration.tif`, with tissue appearing overexposed. Checked actual pixel
  distributions (percentiles) for both `s12t` and `s12q` — they matched
  closely (both go up to ~65500, same percentile profile); `resample()`
  itself (`ClearMap/Alignment/Resampling.py`) does a plain `cv2.resize`,
  never rescales/clips. Root cause: `registration.tif` carries ImageJ
  metadata (`min:0, max:10698`) giving it a sane default display window;
  `resampled.tif` (written by ClearMap's tifffile writer) has none, so
  ImageJ falls back to an unhelpful auto-contrast (often based on just the
  currently-displayed slice) that can make real signal look blown out. **Not
  a real bug** — just a display-window gotcha. Fix if it keeps being
  confusing: write ImageJ min/max metadata when saving `resampled.tif`, or
  just manually set B&C range (~0-8000) in ImageJ.
- **First real registration attempt crashed**: `elastix.log` showed
  `itk::ERROR: RandomCoordinateSampler: Could not find enough image samples
  within reasonable time. Probably the mask is too small` — the `-fMask`
  from `BrainMask.py` was too sparse (9.9% coverage) for
  `ImageSampler "RandomCoordinate"`'s rejection-sampling to work, so the
  affine stage errored out before writing any `TransformParameters.*.txt`,
  which later crashed `process_cell_class` -> `transformation()` ->
  `elx.transform_points()` with "Cannot find a valid transformation file."
  - Root cause of the sparse mask: signal is concentrated in cortex, so a
    plain Otsu + 3D `binary_fill_holes` mask was a hollow cortical **shell**,
    not a solid brain — see [[brain_mask_convex_hull]] (Claude memory) for
    the full note. **Fixed** in `ClearMap/Alignment/BrainMask.py`:
    per-Z-slice `skimage.morphology.convex_hull_image` instead of 3D
    fill_holes (connects each coronal section's outer outline, fills the
    interior). Coverage went 9.9% -> 23.4% on `s12t` (~45% fill within the
    mask's own bounding box — anatomically plausible for a tapered brain).
    **User visually confirmed this mask looks correct** in ImageJ.
  - Did **not** yet switch `ImageSampler` to `RandomSparseMask` (the
    elastix-recommended fix for sparse masks) or `ErodeMask` to `false` in
    `align_affine.txt` — holding off since the mask-quality fix alone may be
    enough; only revisit if the sampling error recurs on a re-run.

**Still pending:** an actual successful end-to-end run on `s12t` with all of
the above fixes in place hasn't happened yet. Next time: re-run
`cellMap.py --config config_12t.yaml`, check whether the `RandomCoordinateSampler`
error is gone now that the mask is solid; if it recurs, switch
`align_affine.txt`'s `ImageSampler` to `"RandomSparseMask"` and set
`ErodeMask` to `"false"`.
