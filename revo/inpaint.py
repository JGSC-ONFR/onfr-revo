"""Relleno de daños (agujeros, roturas, arañazos gruesos, zonas marcadas a mano).

- Motas diminutas: Telea (interpolación suave desde el borde).
- Daños mayores fuera de las caras:
    · LaMa (IA de relleno), si su modelo está en models/. Reconstruye fondo,
      ropa o paredes de forma creíble; lo que había debajo no se recupera.
    · Si no, reconstrucción selectiva en frecuencia (FSR, OpenCV xphoto):
      continúa la textura de alrededor, más borrosa.
- Sobre una cara: siempre relleno liso (Telea), nunca IA. Si falta
  información en un ojo o una boca, queda borroso en lugar de inventado.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .faces import MODELS_DIR

SMALL = 60  # píxeles: por debajo, una mota
LAMA = "lama_fp32.onnx"
LAMA_MIN = 1500  # píxeles: por debajo, el relleno clásico basta
BUDGET = 9_000  # píxeles del recorte reducido que procesa FSR (tiempo acotado)
_lama = None


def lama_available(models_dir: str = MODELS_DIR) -> bool:
    if not os.path.exists(os.path.join(models_dir, LAMA)):
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _lama_session():
    global _lama
    if _lama is None:
        import onnxruntime as ort

        _lama = ort.InferenceSession(os.path.join(MODELS_DIR, LAMA), providers=["CPUExecutionProvider"])
    return _lama


def _lama_fill(crop: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """LaMa trabaja a 512x512: imagen en [0,1] (1,3,512,512) y máscara
    (1,1,512,512) con 1 = hueco."""
    sess = _lama_session()
    h, w = crop.shape[:2]
    img = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    m = (cv2.resize(hole.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    inputs = sess.get_inputs()
    feed = {}
    for inp in inputs:
        if "mask" in inp.name.lower() or (inp.shape and inp.shape[1] == 1):
            feed[inp.name] = m[None, None]
        else:
            feed[inp.name] = img.transpose(2, 0, 1)[None]
    out = sess.run(None, feed)[0][0].transpose(1, 2, 0)
    if out.max() <= 1.5:
        out = out * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


def _fsr(crop: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """FSR «best» sobre una copia reducida (≤ BUDGET px): más natural que la
    versión rápida y con un tiempo acotado."""
    h, w = crop.shape[:2]
    if hole.sum() < 1500:  # daño pequeño: la versión rápida basta
        known = np.where(hole > 0, 0, 255).astype(np.uint8)
        dst = np.zeros_like(crop)
        cv2.xphoto.inpaint(crop, known, dst, cv2.xphoto.INPAINT_FSR_FAST)
        return dst
    f = min(1.0, (BUDGET / (h * w)) ** 0.5)
    small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else crop
    hs = cv2.resize(hole.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) if f < 1 else hole
    known = np.where(hs > 0, 0, 255).astype(np.uint8)
    dst = np.zeros_like(small)
    cv2.xphoto.inpaint(small, known, dst, cv2.xphoto.INPAINT_FSR_BEST)
    return cv2.resize(dst, (w, h), interpolation=cv2.INTER_CUBIC) if f < 1 else dst


def fill(rgb: np.ndarray, mask: np.ndarray, face_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Rellena los píxeles de `mask` (>0). Devuelve (imagen, informe)."""
    out = rgb.copy()
    info = {"small": 0, "large": 0, "on_faces": 0, "engine": "LaMa" if lama_available() else "FSR"}
    if not mask.any():
        return out, info
    m = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    h, w = m.shape
    on_face = (face_mask > 0.05) if face_mask is not None else np.zeros_like(m, bool)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = np.zeros_like(m)
    smooth = np.zeros_like(m)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        comp = lab == i
        if area <= SMALL:
            small[comp] = 255
            info["small"] += 1
            continue
        face_part = comp & on_face
        if face_part.any():
            smooth[face_part] = 255
            info["on_faces"] += 1
        rest = comp & ~on_face
        if not rest.any():
            continue
        info["large"] += 1
        pad = max(32, max(ww, hh))
        y0, y1 = max(0, y - pad), min(h, y + hh + pad)
        x0, x1 = max(0, x - pad), min(w, x + ww + pad)
        # la zona de la cara se trata como hueco también, para que FSR no
        # copie textura de la cara hacia fuera ni al revés
        hole = (rest | face_part)[y0:y1, x0:x1].astype(np.uint8)
        # LaMa solo para daños grandes (≈2,5 s por llamada); los pequeños se
        # rellenan bien y en milésimas con FSR
        use_lama = info["engine"] == "LaMa" and rest.sum() >= LAMA_MIN
        rec = _lama_fill(out[y0:y1, x0:x1], hole) if use_lama else _fsr(out[y0:y1, x0:x1], hole)
        sel = rest[y0:y1, x0:x1]
        out[y0:y1, x0:x1][sel] = rec[sel]

    lisa = ((small | smooth) > 0).astype(np.uint8)
    if lisa.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(lisa, np.ones((15, 15), np.uint8)), connectivity=8)
        for i in range(1, n):
            x, y, ww, hh, _ = stats[i]
            ys, xs = slice(max(0, y - 8), min(h, y + hh + 8)), slice(max(0, x - 8), min(w, x + ww + 8))
            sub = lisa[ys, xs]
            if sub.any():
                rec = cv2.inpaint(out[ys, xs], sub * 255, 3, cv2.INPAINT_TELEA)
                out[ys, xs][sub > 0] = rec[sub > 0]
    return out, info
