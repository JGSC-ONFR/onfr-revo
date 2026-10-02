"""Reconstrucción por simetría: lo que falta de un lado de la cara se toma
del otro lado, que está sano, del propio cuadro.

No se genera nada: se refleja la parte sana alineada con los 68 puntos de la
cara, se iguala la luz y el tono con lo que rodea a la pérdida y solo se pone
dentro de la pérdida. Lo que cae sobre el eje de la cara (centro de la nariz,
de la boca) o tiene dañados los dos lados no se puede reflejar y se deja al
relleno liso de siempre.
"""
from __future__ import annotations

import cv2
import numpy as np

# pareja de cada punto al reflejar la cara (orden de dlib, 68 puntos)
_PAIRS = {i: 16 - i for i in range(17)}
_PAIRS.update({17 + k: 26 - k for k in range(5)})
_PAIRS.update({26 - k: 17 + k for k in range(5)})
_PAIRS.update({27: 27, 28: 28, 29: 29, 30: 30, 31: 35, 32: 34, 33: 33, 34: 32, 35: 31})
_PAIRS.update({36: 45, 37: 44, 38: 43, 39: 42, 40: 47, 41: 46, 45: 36, 44: 37, 43: 38, 42: 39, 47: 40, 46: 41})
_PAIRS.update({48: 54, 49: 53, 50: 52, 51: 51, 52: 50, 53: 49, 54: 48, 55: 59, 56: 58, 57: 57, 58: 56, 59: 55})
_PAIRS.update({60: 64, 61: 63, 62: 62, 63: 61, 64: 60, 65: 67, 66: 66, 67: 65})
MIRROR = np.array([_PAIRS[i] for i in range(68)])


def mirror_map(landmarks: np.ndarray, hole: np.ndarray | None = None) -> np.ndarray | None:
    """Transformación afín que lleva cada punto de la cara a su simétrico.
    Los puntos que caen en la pérdida no cuentan (no son fiables)."""
    pts = landmarks.astype(np.float32)
    ok = np.ones(68, bool)
    if hole is not None:
        h, w = hole.shape
        for i, (x, y) in enumerate(pts):
            xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            ok[i] = not hole[yi, xi]
        ok &= ok[MIRROR]
    if ok.sum() < 12:
        return None
    A, inl = cv2.estimateAffine2D(pts[ok], pts[MIRROR][ok], method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None or (inl is not None and inl.sum() < 10):
        return None
    if np.linalg.det(A[:, :2]) > -0.5:  # tiene que ser un reflejo
        return None
    return A


def mirror_fill(img: np.ndarray, hole: np.ndarray, face_mask: np.ndarray, landmarks: np.ndarray,
                interocular: float) -> tuple[np.ndarray, np.ndarray]:
    """Rellena la parte de `hole` que cae en la cara con su lado simétrico.
    Devuelve (imagen, máscara de lo reconstruido)."""
    h, w = hole.shape
    holeb = hole > 0
    target = holeb & (face_mask > 0.3)
    done = np.zeros((h, w), bool)
    if not target.any():
        return img, done
    A = mirror_map(landmarks, holeb)
    if A is None:
        return img, done
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    mirrored = cv2.warpAffine(img.astype(np.float32), A, (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT)
    src_hole = cv2.warpAffine(holeb.astype(np.uint8), A, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                              borderValue=1) > 0
    src_face = cv2.warpAffine((face_mask > 0.3).astype(np.uint8), A, (w, h),
                              flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0) > 0
    # el reflejo tiene que venir de un sitio sano y lo bastante lejos: en el
    # eje de la cara un punto se refleja sobre sí mismo
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = A[0, 0] * xx + A[0, 1] * yy + A[0, 2]
    sy = A[1, 0] * xx + A[1, 1] * yy + A[1, 2]
    far = np.hypot(sx - xx, sy - yy) > 0.12 * interocular
    usable = target & ~cv2.dilate(src_hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & src_face & far
    if usable.sum() < 20:
        return img, done
    # luz y tono del sitio: diferencia suave entre el original y el reflejo
    # medida en la pintura sana que rodea a la pérdida
    s = max(2.0, 0.25 * interocular)
    ring = (cv2.dilate(holeb.astype(np.uint8), np.ones((int(2 * s) | 1,) * 2, np.uint8)) > 0) & ~holeb & ~src_hole
    ring &= face_mask > 0.3
    if ring.sum() < 15:
        return img, done
    diff = (img.astype(np.float32) - mirrored) * ring[..., None]
    wv = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), s) + 1e-4
    corr = np.stack([cv2.GaussianBlur(diff[..., c], (0, 0), s) for c in range(3)], -1) / wv[..., None]
    # donde no hay referencia cerca, la corrección media del anillo
    mean_corr = diff[ring].mean(0)
    far_w = np.clip(wv / 0.15, 0, 1)[..., None]
    corr = corr * far_w + mean_corr * (1 - far_w)
    patch = np.clip(mirrored + corr, 0, 255)
    out = img.copy()
    out[usable] = patch[usable].astype(np.uint8)
    return out, usable
