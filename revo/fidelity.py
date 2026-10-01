"""Fidelity: verificación de que una cara procesada sigue siendo la misma cara.

Para cada rostro se comparan original y resultado (a la resolución original):
  1. Identidad: distancia entre vectores de reconocimiento facial de dlib.
     (Como referencia, dlib considera "otra persona" a partir de 0.6; aquí
     exigimos mucho menos.)
  2. Geometría y expresión: desplazamiento medio de los 68 puntos faciales,
     relativo a la distancia entre ojos.
  3. Estructura: diferencia de baja frecuencia dentro de la cara frente a la
     versión mínima (solo tono global). Detecta formas nuevas o desplazadas.

Si alguna prueba falla, el pipeline reduce la intervención sobre esa cara y,
en último caso, deja la cara tal como estaba (solo escalada, algo más suave):
mejor un poco borroso que inventado.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .faces import Face, FaceGuard

MIN_IDENTITY, MAX_IDENTITY = 0.08, 0.15
MAX_LANDMARKS = 0.03
MAX_STRUCTURE = 3.0


@dataclass
class FaceCheck:
    identity: float | None
    landmarks: float
    structure: float
    identity_limit: float = MIN_IDENTITY

    @property
    def passed(self) -> bool:
        ok_id = self.identity is None or self.identity <= self.identity_limit
        return ok_id and self.landmarks <= MAX_LANDMARKS and self.structure <= MAX_STRUCTURE


def identity_limit(guard: FaceGuard, face: Face, reference: np.ndarray, noise: float) -> float:
    """Límite de cambio de identidad adaptado a cada cara: 1,5 veces lo que
    varía su vector de identidad si la misma foto se escanease de nuevo (otro
    grano del mismo nivel). Acotado a [0,08 ; 0,15]; dlib separa personas
    distintas a partir de ~0,6."""
    if face.embedding is None:
        return MIN_IDENTITY
    rng = np.random.default_rng(0)
    sigma = max(3.0, noise)
    ds = []
    for _ in range(2):
        noisy = np.clip(reference + rng.normal(0, sigma, reference.shape), 0, 255).astype(np.uint8)
        e = guard.embedding(noisy, face.rect)
        if e is not None:
            ds.append(float(np.linalg.norm(e - face.embedding)))
    jitter = float(np.mean(ds)) if ds else 0.0
    return float(np.clip(1.5 * jitter, MIN_IDENTITY, MAX_IDENTITY))


def _lowpass_L(rgb, sigma):
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    return cv2.GaussianBlur(L, (0, 0), sigma)


def check_face(
    guard: FaceGuard,
    face: Face,
    candidate: np.ndarray,
    reference: np.ndarray,
    face_mask: np.ndarray,
    ignore_mask: np.ndarray | None = None,
    id_limit: float = MIN_IDENTITY,
) -> FaceCheck:
    """candidate y reference a la resolución original."""
    lm = guard.landmarks(candidate, face.rect)
    lm_dev = float(np.linalg.norm(lm - face.landmarks, axis=1).mean() / face.interocular)

    ident = None
    if face.embedding is not None:
        emb = guard.embedding(candidate, face.rect)
        if emb is not None:
            ident = float(np.linalg.norm(emb - face.embedding))

    x0, y0, x1, y1 = face.rect
    pad = int(0.4 * (x1 - x0))
    ys = slice(max(0, y0 - pad), y1 + pad)
    xs = slice(max(0, x0 - pad), x1 + pad)
    sigma = max(1.0, face.interocular / 10)
    a = _lowpass_L(candidate[ys, xs], sigma)
    b = _lowpass_L(reference[ys, xs], sigma)
    w = face_mask[ys, xs].copy()
    if ignore_mask is not None:
        w *= (ignore_mask[ys, xs] == 0)
    struct = float((np.abs(a - b) * w).sum() / max(1e-6, w.sum()))
    return FaceCheck(ident, lm_dev, struct, id_limit)
