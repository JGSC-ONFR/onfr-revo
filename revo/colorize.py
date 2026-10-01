"""Colorización opcional (estimación).

Modelo: Zhang, Isola y Efros, «Colorful Image Colorization» (ECCV 2016),
ejecutado con OpenCV DNN, sin internet una vez descargado.

Regla de fidelidad: solo se añaden los canales de color (a, b). La
luminosidad del resultado es exactamente la de la foto restaurada, así que
ningún contorno, rasgo ni expresión puede cambiar. Los colores son una
estimación del modelo, no información recuperada de la foto.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .faces import MODELS_DIR

PROTO = "colorization_deploy_v2.prototxt"
WEIGHTS = "colorization_release_v2.caffemodel"
POINTS = "pts_in_hull.npy"

_net = None


def available(models_dir: str = MODELS_DIR) -> bool:
    return all(os.path.exists(os.path.join(models_dir, f)) for f in (PROTO, WEIGHTS, POINTS))


def _load(models_dir: str = MODELS_DIR, weights: bool = True):
    global _net
    if _net is None:
        proto = os.path.join(models_dir, PROTO)
        net = cv2.dnn.readNetFromCaffe(proto, os.path.join(models_dir, WEIGHTS)) if weights else cv2.dnn.readNetFromCaffe(proto)
        pts = np.load(os.path.join(models_dir, POINTS)).transpose().reshape(2, 313, 1, 1).astype(np.float32)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1, 313), 2.606, np.float32)]
        _net = net
    return _net


def colorize(rgb: np.ndarray, amount: float = 1.0, net=None) -> np.ndarray:
    """Devuelve la imagen con color estimado. `amount` (0-1) regula la
    intensidad del color. La luminosidad no cambia."""
    net = net or _load()
    lab_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L_full = lab_full[..., 0]

    # el modelo trabaja con L en escala 0-100 a 224x224
    f = rgb.astype(np.float32) / 255.0
    L = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)[..., 0]
    small = cv2.resize(L, (224, 224), interpolation=cv2.INTER_AREA) - 50
    net.setInput(cv2.dnn.blobFromImage(small))
    ab = net.forward()[0].transpose((1, 2, 0))  # 56x56x2, unidades Lab
    ab = cv2.resize(ab, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    ab = ab * float(np.clip(amount, 0, 1.5))

    out = np.empty_like(lab_full)
    out[..., 0] = L_full  # luminosidad intacta
    out[..., 1:] = np.clip(ab + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
