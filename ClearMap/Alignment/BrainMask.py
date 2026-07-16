"""
BrainMask
=========

Otsu-threshold based tissue/brain mask generation, for use as elastix's
``-fMask`` (or ``-mMask``) during registration. Useful when the fixed and
moving images differ a lot in shape/background extent (e.g. small P5 mouse
samples vs. an adult-proportioned atlas), since it keeps the intensity-based
metric (Mattes MI) from being diluted by background voxels.

Also returns the mask's bounding box, which is handy for suggesting
``crop_for_registration`` bounds in config.yaml (see cellMap.py).
"""
import argparse

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import ball

import ClearMap.IO.IO as io


def generate_brain_mask(image, sink_path=None, sigma=1.0, closing_radius=3,
                         dilate_radius=2):
    """Generate a binary tissue mask from a resampled brain image.

    Arguments
    ---------
    image : str or array
        Path to a tif (or any ClearMap-readable source), or an (X,Y,Z) array.
    sink_path : str or None
        If given, write the mask (uint8, 0/1) to this path.
    sigma : float
        Gaussian pre-smoothing sigma (voxels) applied before Otsu
        thresholding, to reduce noise-driven false positives.
    closing_radius : int
        Radius (voxels) of the structuring element used for binary closing
        (bridges small gaps in the thresholded tissue).
    dilate_radius : int
        Radius (voxels) of a final binary dilation, giving the mask a small
        safety margin so it doesn't clip real tissue near the boundary.

    Returns
    -------
    mask : array
        Binary mask, same shape as the input, dtype uint8 (0/1).
    bbox : tuple
        ((x0, x1), (y0, y1), (z0, z1)) bounding box of the mask's nonzero
        extent, in voxels of the input image.
    """
    data = io.read(image) if isinstance(image, str) else np.asarray(image)
    data = data.astype(np.float32)

    smoothed = ndimage.gaussian_filter(data, sigma=sigma)
    mask = smoothed > threshold_otsu(smoothed)

    if closing_radius > 0:
        mask = ndimage.binary_closing(mask, structure=ball(closing_radius))
    mask = ndimage.binary_fill_holes(mask)

    labeled, n_components = ndimage.label(mask)
    if n_components > 1:
        sizes = ndimage.sum(mask, labeled, index=range(1, n_components + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

    if dilate_radius > 0:
        mask = ndimage.binary_dilation(mask, structure=ball(dilate_radius))

    mask = mask.astype(np.uint8)
    bbox = _bounding_box(mask)

    if sink_path is not None:
        io.write(sink_path, mask)
        print(f"Wrote brain mask to {sink_path}: shape={mask.shape}, "
              f"coverage={mask.mean():.1%}, bbox={bbox}")

    return mask, bbox


def _bounding_box(mask):
    """((x0,x1), (y0,y1), (z0,z1)) of the mask's nonzero extent."""
    nonzero = np.argwhere(mask)
    if nonzero.size == 0:
        return tuple((0, s) for s in mask.shape)
    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0) + 1
    return tuple((int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def suggest_crop(bbox, shape, padding=10):
    """Pad `bbox` by `padding` voxels per side, clipped to `shape` — a
    ready-to-paste suggestion for config.yaml's crop_for_registration."""
    return [[max(0, lo - padding), min(size, hi + padding)]
            for (lo, hi), size in zip(bbox, shape)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a binary brain/tissue mask from a resampled '
                     'lightsheet image (Otsu threshold + morphology cleanup).')
    parser.add_argument('input', help='Path to input tif (e.g. resampled.tif)')
    parser.add_argument('output', help='Path to write the output mask tif')
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--closing-radius', type=int, default=3)
    parser.add_argument('--dilate-radius', type=int, default=2)
    parser.add_argument('--padding', type=int, default=10,
                         help='Padding (voxels) for the suggested crop_for_registration bounds')
    args = parser.parse_args()

    mask, bbox = generate_brain_mask(
        args.input, sink_path=args.output, sigma=args.sigma,
        closing_radius=args.closing_radius, dilate_radius=args.dilate_radius)

    suggestion = suggest_crop(bbox, mask.shape, padding=args.padding)
    print(f"\nSuggested crop_for_registration (config.yaml), padded by "
          f"{args.padding} voxels and clipped to image bounds:")
    print(f"  x: {suggestion[0]}")
    print(f"  y: {suggestion[1]}")
    print(f"  z: {suggestion[2]}")
