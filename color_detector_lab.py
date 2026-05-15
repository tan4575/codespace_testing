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
    lab_center: tuple[float, float, float]  # (L*, a*, b*)
    tolerance: tuple[float, float, float]  # per-channel tolerance
    hex_display: str  # representative hex for UI


KL = 1.0  # Lightness weighting
KC = 1.0  # Chroma (a*) weighting
KH = 1.0  # Hue    (b*) weighting

# Full-spectrum LAB color database.
# Centers: real sRGB→LAB conversions (D65 illuminant, rounded to nearest int).
# Tolerances are per-channel LAB units; scaled by KL/KC/KH for tuning.
COLOR_PROFILES: list[ColorProfile] = [
    # ── Reds ───────────────────────────────────────────────────────────────
    ColorProfile("Red", (53, 80, 67), (18 * KL, 22 * KC, 22 * KH), "#FF0000"),
    ColorProfile("Crimson", (41, 64, 28), (15 * KL, 20 * KC, 20 * KH), "#DC143C"),
    ColorProfile("Scarlet", (51, 74, 63), (15 * KL, 20 * KC, 22 * KH), "#FF2400"),
    ColorProfile("Dark Red", (23, 40, 22), (13 * KL, 18 * KC, 18 * KH), "#8B0000"),
    ColorProfile("Maroon", (23, 39, 21), (13 * KL, 18 * KC, 18 * KH), "#800000"),
    ColorProfile("Coral", (61, 43, 37), (16 * KL, 22 * KC, 22 * KH), "#FF6347"),
    ColorProfile("Salmon", (64, 37, 24), (16 * KL, 22 * KC, 20 * KH), "#FA8072"),
    ColorProfile("Tomato", (57, 60, 52), (15 * KL, 20 * KC, 22 * KH), "#FF4500"),
    ColorProfile("Rose", (52, 80, -10), (16 * KL, 22 * KC, 20 * KH), "#FF007F"),
    # ── Oranges ────────────────────────────────────────────────────────────
    ColorProfile("Orange", (72, 24, 69), (18 * KL, 22 * KC, 22 * KH), "#FFA500"),
    ColorProfile("Dark Orange", (66, 36, 73), (16 * KL, 20 * KC, 22 * KH), "#FF8C00"),
    ColorProfile("Amber", (81, 11, 75), (16 * KL, 20 * KC, 22 * KH), "#FFBF00"),
    ColorProfile("Peach", (78, 22, 28), (16 * KL, 20 * KC, 20 * KH), "#FFCBA4"),
    # ── Yellows ────────────────────────────────────────────────────────────
    ColorProfile("Yellow", (97, -22, 95), (12 * KL, 18 * KC, 18 * KH), "#FFFF00"),
    ColorProfile("Lemon", (96, -18, 88), (12 * KL, 18 * KC, 18 * KH), "#FFF44F"),
    ColorProfile("Gold", (84, 3, 79), (16 * KL, 18 * KC, 20 * KH), "#FFD700"),
    ColorProfile("Khaki", (90, -6, 38), (16 * KL, 16 * KC, 20 * KH), "#F0E68C"),
    # ── Yellow-Greens ──────────────────────────────────────────────────────
    ColorProfile("Chartreuse", (90, -50, 77), (16 * KL, 22 * KC, 22 * KH), "#7FFF00"),
    ColorProfile("Yellow Green", (78, -30, 60), (16 * KL, 22 * KC, 22 * KH), "#9ACD32"),
    ColorProfile("Lime", (88, -86, 83), (14 * KL, 20 * KC, 20 * KH), "#00FF00"),
    # ── Greens ─────────────────────────────────────────────────────────────
    ColorProfile("Green", (46, -48, 48), (18 * KL, 22 * KC, 22 * KH), "#008000"),
    ColorProfile("Forest Green", (43, -42, 37), (16 * KL, 20 * KC, 20 * KH), "#228B22"),
    ColorProfile("Dark Green", (32, -33, 28), (14 * KL, 18 * KC, 18 * KH), "#006400"),
    ColorProfile("Emerald", (72, -48, 30), (16 * KL, 22 * KC, 20 * KH), "#50C878"),
    ColorProfile("Olive", (52, -13, 47), (18 * KL, 18 * KC, 22 * KH), "#808000"),
    ColorProfile("Mint", (96, -30, 17), (12 * KL, 18 * KC, 18 * KH), "#98FF98"),
    ColorProfile("Sage", (68, -18, 18), (16 * KL, 18 * KC, 18 * KH), "#B2AC88"),
    # ── Teals ──────────────────────────────────────────────────────────────
    ColorProfile("Teal", (48, -29, -8), (18 * KL, 20 * KC, 20 * KH), "#008080"),
    ColorProfile("Dark Teal", (33, -22, -8), (14 * KL, 18 * KC, 18 * KH), "#005F5F"),
    ColorProfile("Turquoise", (83, -27, -8), (16 * KL, 20 * KC, 20 * KH), "#40E0D0"),
    ColorProfile("Aquamarine", (91, -36, 3), (14 * KL, 18 * KC, 18 * KH), "#7FFFD4"),
    # ── Cyans ──────────────────────────────────────────────────────────────
    ColorProfile("Cyan", (91, -48, -14), (14 * KL, 20 * KC, 20 * KH), "#00FFFF"),
    ColorProfile("Sky Blue", (78, -7, -25), (18 * KL, 18 * KC, 22 * KH), "#87CEEB"),
    # ── Blues ──────────────────────────────────────────────────────────────
    ColorProfile("Cornflower", (60, 9, -53), (16 * KL, 20 * KC, 22 * KH), "#6495ED"),
    ColorProfile("Steel Blue", (48, 2, -40), (16 * KL, 18 * KC, 22 * KH), "#4682B4"),
    ColorProfile("Blue", (32, 79, -108), (18 * KL, 24 * KC, 24 * KH), "#0000FF"),
    ColorProfile("Royal Blue", (40, 24, -69), (16 * KL, 20 * KC, 24 * KH), "#4169E1"),
    ColorProfile("Cobalt", (32, 22, -72), (14 * KL, 18 * KC, 22 * KH), "#0047AB"),
    ColorProfile("Navy", (13, 47, -64), (13 * KL, 18 * KC, 20 * KH), "#000080"),
    ColorProfile(
        "Midnight Blue", (16, 24, -49), (13 * KL, 18 * KC, 20 * KH), "#191970"
    ),
    # ── Violets / Purples ──────────────────────────────────────────────────
    ColorProfile("Indigo", (15, 37, -63), (13 * KL, 20 * KC, 22 * KH), "#4B0082"),
    ColorProfile("Violet", (40, 55, -65), (16 * KL, 22 * KC, 24 * KH), "#7F00FF"),
    ColorProfile("Purple", (30, 50, -38), (16 * KL, 22 * KC, 22 * KH), "#800080"),
    ColorProfile("Dark Purple", (22, 36, -28), (13 * KL, 18 * KC, 18 * KH), "#4B0050"),
    ColorProfile("Orchid", (60, 44, -25), (16 * KL, 20 * KC, 20 * KH), "#DA70D6"),
    ColorProfile("Lavender", (91, 5, -13), (14 * KL, 14 * KC, 16 * KH), "#E6E6FA"),
    ColorProfile("Lilac", (72, 20, -18), (16 * KL, 18 * KC, 18 * KH), "#C8A2C8"),
    ColorProfile("Plum", (41, 37, -22), (14 * KL, 18 * KC, 18 * KH), "#8B4789"),
    # ── Pinks / Magentas ───────────────────────────────────────────────────
    ColorProfile("Magenta", (60, 98, -61), (16 * KL, 22 * KC, 22 * KH), "#FF00FF"),
    ColorProfile("Fuchsia", (55, 85, -45), (15 * KL, 20 * KC, 20 * KH), "#FF1493"),
    ColorProfile("Hot Pink", (64, 64, -17), (16 * KL, 20 * KC, 18 * KH), "#FF69B4"),
    ColorProfile("Pink", (81, 24, 4), (16 * KL, 20 * KC, 16 * KH), "#FFC0CB"),
    ColorProfile("Light Pink", (80, 22, 3), (14 * KL, 18 * KC, 14 * KH), "#FFB6C1"),
    # ── Neutrals ───────────────────────────────────────────────────────────
    ColorProfile("White", (100, 0, 0), (6 * KL, 8 * KC, 8 * KH), "#FFFFFF"),
    ColorProfile("Ivory", (99, -2, 8), (6 * KL, 8 * KC, 10 * KH), "#FFFFF0"),
    ColorProfile("Cream", (96, -3, 10), (8 * KL, 8 * KC, 10 * KH), "#FFFDD0"),
    ColorProfile("Light Gray", (83, 0, 0), (10 * KL, 8 * KC, 8 * KH), "#D3D3D3"),
    ColorProfile("Silver", (77, 0, 0), (10 * KL, 8 * KC, 8 * KH), "#C0C0C0"),
    ColorProfile("Gray", (53, 0, 0), (12 * KL, 8 * KC, 8 * KH), "#808080"),
    ColorProfile("Dark Gray", (28, 0, 0), (10 * KL, 8 * KC, 8 * KH), "#404040"),
    ColorProfile("Charcoal", (22, -2, -3), (10 * KL, 8 * KC, 8 * KH), "#36454F"),
    ColorProfile("Black", (0, 0, 0), (8 * KL, 8 * KC, 8 * KH), "#000000"),
    # ── Earth Tones ────────────────────────────────────────────────────────
    ColorProfile("Brown", (36, 22, 29), (16 * KL, 20 * KC, 22 * KH), "#8B4513"),
    ColorProfile("Dark Brown", (27, 16, 22), (13 * KL, 16 * KC, 18 * KH), "#5C3317"),
    ColorProfile("Chocolate", (50, 26, 39), (16 * KL, 20 * KC, 22 * KH), "#D2691E"),
    ColorProfile("Sienna", (44, 26, 35), (16 * KL, 20 * KC, 22 * KH), "#A0522D"),
    ColorProfile("Tan", (73, 9, 23), (16 * KL, 16 * KC, 18 * KH), "#D2B48C"),
    ColorProfile("Sandy Brown", (70, 14, 36), (16 * KL, 18 * KC, 20 * KH), "#F4A460"),
    ColorProfile("Beige", (96, -2, 11), (10 * KL, 10 * KC, 12 * KH), "#F5F5DC"),
    ColorProfile("Rust", (42, 37, 40), (16 * KL, 20 * KC, 20 * KH), "#B7410E"),
    ColorProfile("Terracotta", (57, 36, 30), (16 * KL, 20 * KC, 20 * KH), "#E2725B"),
    ColorProfile("Ochre", (63, 12, 52), (16 * KL, 18 * KC, 22 * KH), "#CC7722"),
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
        self.calibrate = False

        # Pre-build color database as numpy arrays for vectorized matching
        self._centers = np.array(
            [p.lab_center for p in COLOR_PROFILES], dtype=np.float32
        )
        self._tolerances = np.array(
            [p.tolerance for p in COLOR_PROFILES], dtype=np.float32
        )

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
        if not self.calibrate:
            self._initial_background_cal(bgr_image)
            self.calibrate = True

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
        # bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        result = self.detect(bgr_image, roi)
        vis = self._draw_overlay(bgr_image.copy(), result, roi)
        return result, vis

    def _initial_background_cal(self, img: np.ndarray):
        # Apply slight blur to reduce high-frequency noise
        self.frame = cv2.GaussianBlur(img, (21, 21), 0)

        # Convert to float32 for accumulateWeighted
        # Grayscale is faster, but this works on BGR too
        self.avg_bg = np.float32(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY))

        # Hyperparameter: Learning rate (0 to 1)
        self.alpha = 0.02

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray, roi: Optional[tuple]) -> np.ndarray:
        if roi:
            x, y, w, h = roi
            img = img[y : y + h, x : x + w]

        # Prepare current frame
        blur = cv2.GaussianBlur(img, (21, 21), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # 2. Update the Background Model
        # This keeps a "running average" of the scene
        cv2.accumulateWeighted(gray, self.avg_bg, self.alpha)

        # 3. Calculate the Absolute Difference
        # Convert background back to uint8 to compare with the current frame
        res_bg = cv2.convertScaleAbs(self.avg_bg)
        diff = cv2.absdiff(gray, res_bg)

        # 4. Thresholding to find moving objects
        # Any pixel difference > 25 is marked as foreground (white)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Optional: Clean up noise with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        thresh = cv2.dilate(thresh, None, iterations=2)
        color_result = cv2.bitwise_and(self.frame, self.frame, mask=thresh)

        return color_result
        # # Gaussian blur to reduce sensor noise & texture artifacts
        # if self.blur_kernel > 1:
        #     k = self.blur_kernel | 1  # ensure odd
        #     img = cv2.GaussianBlur(img, (k, k), 0)

        # # Morphological opening removes small noise blobs
        # kernel = cv2.getStructuringElement(
        #     cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        # )
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # return img

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

        # Prefer chromatic pixels over neutral ones so that white/black backgrounds
        # and achromatic car parts (roof, tires, trim) don't swamp the vote.
        # OpenCV LAB: L in [0,255], a/b shifted by 128.

        # Subsample for speed on large images (max 8 000 pixels)
        # if len(pixels) > 8_000:
        #     idx = np.random.choice(len(pixels), 8_000, replace=False)
        #     pixels = pixels[idx]

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,  # max iterations
            0.3,  # epsilon
        )
        _, labels, centers = cv2.kmeans(
            pixels, self.k, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
        )

        # Largest cluster wins; pixels are already pre-filtered to be chromatic
        # so a plain count is a reliable proxy for the dominant hue.
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
    def _calculate_delta_e_cie76(self, lab1, lab2):
        """
        Calculates Delta E (CIE76) as the Euclidean distance between
        two points in the 3D Lab space.
        """
        # Formula: sqrt((L2-L1)^2 + (a2-a1)^2 + (b2-b1)^2)
        return np.sqrt(np.sum((lab1 - lab2) ** 2))

    def _match_color(self, lab: np.ndarray) -> dict:
        """
        Match a LAB value against the color database using a
        tolerance-weighted Euclidean distance (perceptual ΔE approximation).
        """
        # Normalised per-channel distance: (obs - center) / tolerance
        RAD = np.float32(np.pi / 180)
        L1 = lab[0]
        a1 = lab[1]
        b1 = lab[2]
        L2 = self._centers[:, 0]
        a2 = self._centers[:, 1]
        b2 = self._centers[:, 2]
        c1 = np.hypot(a1, b1)
        c2 = np.hypot(a2, b2)  # (N,) — sample chroma
        Cavg = (c1 + c2) * np.float32(0.5)
        Cavg7 = Cavg**7
        G = np.float32(0.5) * (1 - np.sqrt(Cavg7 / (Cavg7 + np.float32(25**7))))

        a1p = a1 * (1 + G)
        a2p = a2 * (1 + G)
        C1p = np.hypot(a1p, b1)
        C2p = np.hypot(a2p, b2)

        def hue_prime(ap, b):
            ang = np.degrees(np.arctan2(b, ap))  # (N,)
            return np.where(ang < 0, ang + 360, ang)  # wrap negatives

        h1p = np.where((a1p == 0) & (b1 == 0), np.float32(0), hue_prime(a1p, b1))
        h2p = np.where((a2p == 0) & (b2 == 0), np.float32(0), hue_prime(a2p, b2))

        dLp = L2 - L1
        dCp = C2p - C1p

        achromatic = (C1p * C2p) == 0
        diff = h2p - h1p
        dhp = np.where(
            achromatic,
            np.float32(0),
            np.where(
                np.abs(diff) <= 180, diff, np.where(diff > 180, diff - 360, diff + 360)
            ),
        )

        dHp = 2 * np.sqrt(C1p * C2p) * np.sin(dhp * np.float32(0.5) * RAD)
        Lbp = (L1 + L2) * np.float32(0.5)
        Cbp = (C1p + C2p) * np.float32(0.5)

        # Branchless mean hue (replaces 4-way if/elif)
        hsum = h1p + h2p
        hbp = np.where(
            achromatic,
            hsum,
            np.where(
                np.abs(h1p - h2p) <= 180,
                hsum * np.float32(0.5),
                np.where(
                    hsum < 360,
                    (hsum + 360) * np.float32(0.5),
                    (hsum - 360) * np.float32(0.5),
                ),
            ),
        )
        hbp_r = hbp * RAD  # degrees → radians (reused 4×)
        T = (
            np.float32(1)
            - np.float32(0.17) * np.cos(hbp_r - np.float32(30 * np.pi / 180))
            + np.float32(0.24) * np.cos(2 * hbp_r)
            + np.float32(0.32) * np.cos(3 * hbp_r + np.float32(6 * np.pi / 180))
            - np.float32(0.20) * np.cos(4 * hbp_r - np.float32(63 * np.pi / 180))
        )

        Lbp50 = Lbp - np.float32(50)
        SL = np.float32(1) + np.float32(0.015) * Lbp50**2 / np.sqrt(
            np.float32(20) + Lbp50**2
        )
        SC = np.float32(1) + np.float32(0.045) * Cbp
        SH = np.float32(1) + np.float32(0.015) * Cbp * T

        Cbp7 = Cbp**7
        RC = np.float32(2) * np.sqrt(Cbp7 / (Cbp7 + np.float32(25**7)))
        dtheta = np.float32(30) * np.exp(
            -(((hbp - np.float32(275)) / np.float32(25)) ** 2)
        )
        RT = -np.sin(np.float32(2) * dtheta * RAD) * RC

        kl, kc, kh = 1.0, 1.0, 1.0
        Lterm = dLp / (kl * SL) / 100
        Cterm = dCp / (kc * SC) / 100
        Hterm = dHp / (kh * SH) / 100
        distances = np.sqrt(Lterm**2 + Cterm**2 + Hterm**2 + RT * Cterm * Hterm)
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
            "lab_value": {
                "L": round(float(lab[0]), 1),
                "a": round(float(lab[1]), 1),
                "b": round(float(lab[2]), 1),
            },
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
            cv2.putText(
                img, line, (10, y_pos), font, 0.65, (255, 255, 255), 1, cv2.LINE_AA
            )

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
    img_parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Region of interest (optional)",
    )
    img_parser.add_argument(
        "--show", action="store_true", help="Display annotated image window"
    )

    # webcam mode
    cam_parser = subparsers.add_parser("webcam", help="Real-time webcam detection")
    cam_parser.add_argument("--camera", type=int, default=0, help="Camera index")

    # shared params
    for p in (img_parser, cam_parser):
        p.add_argument(
            "--clusters", type=int, default=4, help="K-Means clusters (default 4)"
        )

    args = parser.parse_args()
    detector = LabColorDetector(k_clusters=args.clusters)

    if args.mode == "image":
        roi = tuple(args.roi) if args.roi else None
        result, vis = detector.detect_with_visualization(
            cv2.imread(args.path),
            roi,  # type: ignore[arg-type]
        )
        print("\n── Color Detection Result ─────────────────")
        print(f"  Detected Color : {result['color']}")
        print(f"  Confidence     : {result['confidence']:.1%}")
        print(
            f"  LAB Value      : L={result['lab_value']['L']}  "
            f"a={result['lab_value']['a']}  b={result['lab_value']['b']}"
        )
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
