"""Decode a video into per-frame image statistics."""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

WIDTH = 96
HEIGHT = 64


def signal(bgr, previous_gray):
    """Return the resized grayscale frame and its four summary statistics.

    ``previous_gray`` is the grayscale frame returned for the preceding video
    frame, or ``None`` for the first frame.
    """
    if bgr is None:
        raise ValueError("frame is None")

    resized = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray /= np.float32(255.0)

    if gray.shape != (HEIGHT, WIDTH):
        raise ValueError(f"unexpected grayscale shape: {gray.shape}")
    if not np.isfinite(gray).all():
        raise ValueError("non-finite value in resized grayscale frame")

    mean = float(np.mean(gray, dtype=np.float64))
    std = float(np.std(gray, dtype=np.float64))

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mean = float(
        np.mean((np.abs(sobel_x) + np.abs(sobel_y)) / np.float32(2.0), dtype=np.float64)
    )

    if previous_gray is None:
        difference = 0.0
    else:
        previous = np.asarray(previous_gray, dtype=np.float32)
        if previous.shape != gray.shape:
            raise ValueError(
                f"previous grayscale shape {previous.shape} does not match {gray.shape}"
            )
        difference = float(np.mean(np.abs(gray - previous), dtype=np.float64))

    stats = (mean, std, sobel_mean, difference)
    if not all(np.isfinite(value) for value in stats):
        raise ValueError("non-finite value in frame statistics")

    return gray, stats


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Decode a video and save per-frame grayscale statistics."
    )
    parser.add_argument("input", help="input video path")
    parser.add_argument("output", help="output .npy path")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    cv2.setNumThreads(1)

    capture = cv2.VideoCapture(args.input)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {args.input}")

        rows = []
        previous_gray = None
        while True:
            ok, bgr = capture.read()
            if not ok:
                break

            previous_gray, stats = signal(bgr, previous_gray)
            rows.append(stats)

        if not rows:
            raise RuntimeError(f"video contains no decodable frames: {args.input}")

        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
            raise RuntimeError(f"unexpected statistics shape: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("non-finite value in output statistics")

        np.save(args.output, values)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        capture.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
