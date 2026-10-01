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
import threading

import cv2
import numpy as np

from .faces import MODELS_DIR

SMALL = 60  # píxeles: por debajo, una mota
LAMA = "lama_fp32.onnx"
LAMA_MIN = 1500  # píxeles: por debajo, el relleno clásico basta
BUDGET = 25_000  # píxeles del recorte reducido que procesa FSR (tiempo acotado)
LAMA_TILE = 1024  # lado máximo de un grupo de daños rellenado en una sola pasada de LaMa
_lama = None
_lama_lock = threading.Lock()


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
    with _lama_lock:  # si la precarga está en marcha, se espera a ella
        if _lama is None:
            import onnxruntime as ort

            _lama = ort.InferenceSession(os.path.join(MODELS_DIR, LAMA), providers=["CPUExecutionProvider"])
    return _lama


def preload() -> None:
    """Carga LaMa en segundo plano (≈13 s en un portátil) para que la
    primera foto no tenga que esperarla."""
    if lama_available():
        threading.Thread(target=_lama_session, daemon=True).start()


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
    """FSR rápido sobre una copia reducida (≤ BUDGET px): su coste crece con
    el cuadrado del tamaño, así que se acota para que nunca pase de ~0,1 s."""
    h, w = crop.shape[:2]
    f = min(1.0, (BUDGET / (h * w)) ** 0.5)
    small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else crop
    hs = cv2.resize(hole.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) if f < 1 else hole
    known = np.where(hs > 0, 0, 255).astype(np.uint8)
    dst = np.zeros_like(small)
    cv2.xphoto.inpaint(np.ascontiguousarray(small), known, dst, cv2.xphoto.INPAINT_FSR_FAST)
    return cv2.resize(dst, (w, h), interpolation=cv2.INTER_CUBIC) if f < 1 else dst


def _clusters(boxes: list[list[int]], limit: int) -> list[list[int]]:
    """Une recuadros cercanos mientras el resultado no pase de `limit` px de
    lado: varios daños juntos se rellenan en una sola pasada."""
    boxes = [b[:] for b in boxes]
    merged = True
    while merged:
        merged = False
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                A, B = boxes[a], boxes[b]
                u = [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])]
                if max(u[2] - u[0], u[3] - u[1]) <= limit:
                    boxes[a] = u
                    del boxes[b]
                    merged = True
                    break
            if merged:
                break
    return boxes


def fill(rgb: np.ndarray, mask: np.ndarray, face_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Rellena los píxeles de `mask` (>0). Devuelve (imagen, informe)."""
    out = rgb.copy()
    info = {"small": 0, "large": 0, "on_faces": 0, "engine": "LaMa" if lama_available() else "FSR", "passes": 0}
    if not mask.any():
        return out, info
    m = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    h, w = m.shape
    on_face = (face_mask > 0.05) if face_mask is not None else np.zeros_like(m, bool)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = np.zeros_like(m)
    smooth = np.zeros_like(m)
    big_rest = np.zeros((h, w), bool)  # se rellena con LaMa
    mid_rest = np.zeros((h, w), bool)  # se rellena con FSR
    big_boxes, mid_boxes = [], []
    on_face_any = on_face.any()
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area <= SMALL:
            info["small"] += 1
            small[y:y + hh, x:x + ww][lab[y:y + hh, x:x + ww] == i] = 255
            continue
        comp = lab[y:y + hh, x:x + ww] == i
        of = on_face[y:y + hh, x:x + ww] if on_face_any else np.zeros_like(comp)
        face_part = comp & of
        if face_part.any():
            smooth[y:y + hh, x:x + ww][face_part] = 255
            info["on_faces"] += 1
        rest = comp & ~of
        if not rest.any():
            continue
        info["large"] += 1
        pad = max(24, int(0.5 * max(ww, hh)))
        box = [max(0, x - pad), max(0, y - pad), min(w, x + ww + pad), min(h, y + hh + pad)]
        if info["engine"] == "LaMa" and rest.sum() >= LAMA_MIN:
            big_rest[y:y + hh, x:x + ww] |= rest
            big_boxes.append(box)
        else:
            mid_rest[y:y + hh, x:x + ww] |= rest
            mid_boxes.append(box)

    # la cara se trata como hueco también, para no copiar textura de la cara
    # hacia fuera ni al revés
    face_hole = smooth > 0
    for rest_mask, boxes, fn, limit in (
        (big_rest, big_boxes, _lama_fill, LAMA_TILE),
        (mid_rest, mid_boxes, _fsr, 400),
    ):
        for x0, y0, x1, y1 in _clusters(boxes, limit):
            sel = rest_mask[y0:y1, x0:x1]
            if not sel.any():
                continue
            hole = (sel | face_hole[y0:y1, x0:x1] | big_rest[y0:y1, x0:x1] | mid_rest[y0:y1, x0:x1]).astype(np.uint8)
            rec = fn(out[y0:y1, x0:x1], hole)
            out[y0:y1, x0:x1][sel] = rec[sel]
            info["passes"] += 1

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
