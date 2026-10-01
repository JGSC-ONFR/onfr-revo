"""Mejora: aumento de resolución y nitidez sin inventar detalle.

El escalado es clásico (Lanczos) y la nitidez solo refuerza bordes que ya
existen, con control de halos: ningún píxel puede salir del rango de sus
vecinos originales. Por diseño, aquí no se genera textura nueva.
"""
from __future__ import annotations

import cv2
import numpy as np


def upscale(rgb: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1:
        return rgb.copy()
    return cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def sharpen(rgb: np.ndarray, amount: float, scale: float, noise: float) -> np.ndarray:
    if amount <= 0:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    sigma = 0.6 + 0.5 * scale
    detail = L - cv2.GaussianBlur(L, (0, 0), sigma)
    # umbral suave: no amplificar lo que es ruido
    thr = max(1.0, noise * 0.8)
    weight = np.clip((np.abs(detail) - thr) / (2 * thr), 0, 1)
    out = L + amount * detail * weight
    # control de halos: limitar al rango local
    k = np.ones((3, 3), np.uint8)
    lo = cv2.erode(L, k) - 2
    hi = cv2.dilate(L, k) + 2
    out = np.clip(out, lo, hi)
    lab[..., 0] = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
