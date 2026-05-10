"""
Color Detection using LAB Color Space
======================================
Optimized OpenCV-based color detection leveraging the perceptually uniform
LAB color space for accurate, robust object color identification.

LAB Channels:
  L* = Lightness (0–100)
  a* = Green (negative) to Red (positive)
  b* = Blue (negative) to Yellow (positive)

Author : Computer Vision Engineer
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import argparse
import sys


# ─────────────────────────────────────────────
# Color Database  (LAB center, tolerance)
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class ColorProfile:
    name: str
    lab_center: tuple[float, float, float]   # (L*, a*, b*)
    tolerance: tuple[float, float, float]    # per-channel tolerance
    hex_display: str                          # representative hex for UI


# Empirically tuned LAB ranges for common object colors
COLOR_PROFILES: list[ColorProfile] = [
    ColorProfile("Red",      (45, 55,  38),  (25, 30, 30), "#E53935"),
    ColorProfile("Orange",   (60, 30,  50),  (20, 25, 30), "#FB8C00"),
    ColorProfile("Yellow",   (85,  0,  75),  (20, 20, 35), "#FDD835"),
    ColorProfile("Green",    (45, -40, 25),  (25, 30, 30), "#43A047"),
    ColorProfile("Cyan",     (70, -25,-20),  (20, 25, 25), "#00ACC1"),
    ColorProfile("Blue",     (35,  10,-50),  (25, 20, 35), "#1E88E5"),
    ColorProfile("Purple",   (35,  30,-30),  (20, 25, 25), "#8E24AA"),
    ColorProfile("Pink",     (70,  35,  5),  (20, 25, 20), "#E91E63"),
    ColorProfile("White",    (95,   0,  0),  (10, 10, 10), "#FAFAFA"),
    ColorProfile("Black",    (10,   0,  0),  (12, 10, 10), "#212121"),
    ColorProfile("Gray",     (55,   0,  0),  (20, 10, 10), "#757575"),
    ColorProfile("Brown",    (35,  15, 20),  (20, 20, 20), "#6D4C41"),
]


# ─────────────────────────────────────────────
# Core Detector
# ─────────────────────────────────────────────
class LabColorDetector:
    """
    Detects the dominant color of objects in an image using the
    perceptually uniform LAB color space.

    Pipeline:
        1. Convert BGR → LAB
        2. Extract region-of-interest (ROI) or full image
        3. K-Means quantization to find dominant LAB cluster
        4. Match cluster centroid against color profile database
        5. Return color name + confidence
    """

    def __init__(
        self,
        k_clusters: int = 4,
        blur_kernel: int = 5,
        morph_kernel: int = 3,
        min_confidence: float = 0.55,
    ):
        self.k = k_clusters
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self.min_confidence = min_confidence

        # Pre-build color database as numpy arrays for vectorized matching
        self._centers = np.array([p.lab_center for p in COLOR_PROFILES], dtype=np.float32)
        self._tolerances = np.array([p.tolerance for p in COLOR_PROFILES], dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_from_file(self, image_path: str) -> dict:
        """Detect dominant color from an image file path."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        return self.detect(img)

    def detect(self, bgr_image: np.ndarray, roi: Optional[tuple] = None) -> dict:
        """
        Detect dominant color in a BGR image.

        Args:
            bgr_image : HxWx3 uint8 BGR image (standard OpenCV format)
            roi       : Optional (x, y, w, h) region of interest

        Returns:
            dict with keys: color, confidence, lab_value, hex_color, all_matches
        """
        img = self._preprocess(bgr_image, roi)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        dominant_lab = self._dominant_lab_cluster(lab_img)
        return self._match_color(dominant_lab)

    def detect_with_visualization(
        self, bgr_image: np.ndarray, roi: Optional[tuple] = None
    ) -> tuple[dict, np.ndarray]:
        """
        Detect color AND return an annotated visualization frame.

        Returns:
            (result_dict, annotated_bgr_image)
        """
        result = self.detect(bgr_image, roi)
        vis = self._draw_overlay(bgr_image.copy(), result, roi)
        return result, vis

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray, roi: Optional[tuple]) -> np.ndarray:
        if roi:
            x, y, w, h = roi
            img = img[y:y + h, x:x + w]

        # Gaussian blur to reduce sensor noise & texture artifacts
        if self.blur_kernel > 1:
            k = self.blur_kernel | 1  # ensure odd
            img = cv2.GaussianBlur(img, (k, k), 0)

        # Morphological opening removes small noise blobs
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return img

    # ------------------------------------------------------------------
    # Dominant color via K-Means in LAB space
    # ------------------------------------------------------------------

    def _dominant_lab_cluster(self, lab_img: np.ndarray) -> np.ndarray:
        """
        Run K-Means on the LAB pixels; return the centroid of the
        largest cluster (dominant color).

        OpenCV K-Means operates on float32 row vectors.
        """
        pixels = lab_img.reshape(-1, 3).astype(np.float32)

        # Subsample for speed on large images (max 8 000 pixels)
        if len(pixels) > 8_000:
            idx = np.random.choice(len(pixels), 8_000, replace=False)
            pixels = pixels[idx]

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,     # max iterations
            1.0,    # epsilon
        )
        _, labels, centers = cv2.kmeans(
            pixels, self.k, None, criteria,
            attempts=3, flags=cv2.KMEANS_PP_CENTERS
        )

        # Find the cluster with the most pixels
        counts = np.bincount(labels.flatten())
        dominant_center = centers[np.argmax(counts)]

        # OpenCV encodes LAB as: L=[0,255], a=[0,255], b=[0,255]
        # Convert back to standard: L=[0,100], a=[-128,127], b=[-128,127]
        L = dominant_center[0] * 100.0 / 255.0
        a = dominant_center[1] - 128.0
        b = dominant_center[2] - 128.0
        return np.array([L, a, b], dtype=np.float32)

    # ------------------------------------------------------------------
    # Color matching
    # ------------------------------------------------------------------

    def _match_color(self, lab: np.ndarray) -> dict:
        """
        Match a LAB value against the color database using a
        tolerance-weighted Euclidean distance (perceptual ΔE approximation).
        """
        # Normalised per-channel distance: (obs - center) / tolerance
        delta = (lab - self._centers) / (self._tolerances + 1e-6)
        distances = np.linalg.norm(delta, axis=1)  # shape (N,)

        sorted_idx = np.argsort(distances)
        best_idx = sorted_idx[0]
        best_dist = distances[best_idx]

        # Confidence: exponential decay — distance=0 → 1.0, distance=3 → ~0.05
        confidence = float(np.exp(-0.6 * best_dist))

        all_matches = [
            {
                "color": COLOR_PROFILES[i].name,
                "confidence": round(float(np.exp(-0.6 * distances[i])), 3),
            }
            for i in sorted_idx[:3]
        ]

        profile = COLOR_PROFILES[best_idx]
        return {
            "color": profile.name if confidence >= self.min_confidence else "Unknown",
            "confidence": round(confidence, 3),
            "lab_value": {"L": round(float(lab[0]), 1),
                          "a": round(float(lab[1]), 1),
                          "b": round(float(lab[2]), 1)},
            "hex_color": profile.hex_display,
            "all_matches": all_matches,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _draw_overlay(
        self, img: np.ndarray, result: dict, roi: Optional[tuple]
    ) -> np.ndarray:
        h, w = img.shape[:2]
        color_name = result["color"]
        confidence = result["confidence"]
        lab = result["lab_value"]

        # ── ROI rectangle ──────────────────────────────────────────────
        if roi:
            x, y, rw, rh = roi
            cv2.rectangle(img, (x, y), (x + rw, y + rh), (0, 255, 200), 2)

        # ── Color swatch (bottom-left corner) ─────────────────────────
        swatch_hex = result["hex_color"].lstrip("#")
        r, g, b = (
            int(swatch_hex[0:2], 16),
            int(swatch_hex[2:4], 16),
            int(swatch_hex[4:6], 16),
        )
        swatch_bgr = (b, g, r)
        cv2.rectangle(img, (10, h - 70), (70, h - 10), swatch_bgr, -1)
        cv2.rectangle(img, (10, h - 70), (70, h - 10), (200, 200, 200), 1)

        # ── Text overlay ───────────────────────────────────────────────
        font = cv2.FONT_HERSHEY_DUPLEX
        lines = [
            f"Color    : {color_name}",
            f"Confidence: {confidence:.0%}",
            f"LAB      : L={lab['L']}  a={lab['a']}  b={lab['b']}",
        ]
        for i, line in enumerate(lines):
            y_pos = 30 + i * 28
            cv2.putText(img, line, (10, y_pos), font, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y_pos), font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        return img


# ─────────────────────────────────────────────
# Live Webcam Demo
# ─────────────────────────────────────────────

def run_webcam_demo(detector: LabColorDetector, camera_index: int = 0) -> None:
    """
    Real-time color detection on the centre 40% ROI of the webcam feed.
    Press  Q  to quit.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("Webcam demo running — press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        roi_x = int(fw * 0.30)
        roi_y = int(fh * 0.30)
        roi_w = int(fw * 0.40)
        roi_h = int(fh * 0.40)
        roi = (roi_x, roi_y, roi_w, roi_h)

        result, vis = detector.detect_with_visualization(frame, roi)
        cv2.imshow("LAB Color Detector  [Q = quit]", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect object color using LAB color space (OpenCV)"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # image mode
    img_parser = subparsers.add_parser("image", help="Analyze a single image file")
    img_parser.add_argument("path", type=str, help="Path to image file")
    img_parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                            help="Region of interest (optional)")
    img_parser.add_argument("--show", action="store_true",
                            help="Display annotated image window")

    # webcam mode
    cam_parser = subparsers.add_parser("webcam", help="Real-time webcam detection")
    cam_parser.add_argument("--camera", type=int, default=0, help="Camera index")

    # shared params
    for p in (img_parser, cam_parser):
        p.add_argument("--clusters", type=int, default=4,
                       help="K-Means clusters (default 4)")

    args = parser.parse_args()
    detector = LabColorDetector(k_clusters=args.clusters)

    if args.mode == "image":
        roi = tuple(args.roi) if args.roi else None
        result, vis = detector.detect_with_visualization(
            cv2.imread(args.path), roi  # type: ignore[arg-type]
        )
        print("\n── Color Detection Result ─────────────────")
        print(f"  Detected Color : {result['color']}")
        print(f"  Confidence     : {result['confidence']:.1%}")
        print(f"  LAB Value      : L={result['lab_value']['L']}  "
              f"a={result['lab_value']['a']}  b={result['lab_value']['b']}")
        print(f"  Hex (approx)   : {result['hex_color']}")
        print(f"  Top matches    : {result['all_matches']}")
        print("────────────────────────────────────────────\n")

        if args.show:
            cv2.imshow("Result", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.mode == "webcam":
        run_webcam_demo(detector, camera_index=args.camera)


if __name__ == "__main__":
    main()
