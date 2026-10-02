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

import atexit
import os
import pickle
import subprocess
import sys
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


def _start_worker():
    """Lanza (una vez) el proceso que carga y ejecuta LaMa."""
    global _lama
    if _lama is None or _lama.poll() is not None:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # sin ventana extra en Windows
        _lama = subprocess.Popen(
            # se ejecuta por ruta, no con -m: así no importa el paquete revo (dlib…)
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lama_worker.py"),
             os.path.join(MODELS_DIR, LAMA)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            creationflags=flags,
        )
        _lama.ready = False
        atexit.register(_lama.kill)
    return _lama


def preload() -> None:
    """Empieza a cargar LaMa en su propio proceso (≈13-30 s en un portátil)
    sin bloquear la interfaz: cuando llegue la primera foto ya estará lista."""
    if lama_available():
        with _lama_lock:
            _start_worker()


def _lama_fill(crop: np.ndarray, hole: np.ndarray) -> np.ndarray:
    with _lama_lock:
        p = _start_worker()
        if not p.ready:
            if pickle.load(p.stdout) != "ready":
                raise RuntimeError("LaMa no ha arrancado")
            p.ready = True
        pickle.dump((np.ascontiguousarray(crop), hole.astype(np.uint8)), p.stdin)
        p.stdin.flush()
        res = pickle.load(p.stdout)
    if isinstance(res, Exception):
        raise res
    return res


def _fsr(crop: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """FSR rápido sobre una copia reducida (≤ BUDGET px): su coste crece con
    el cuadrado del tamaño, así que se acota para que nunca pase de ~0,1 s."""
    h, w = crop.shape[:2]
    f = min(1.0, (BUDGET / (h * w)) ** 0.5)
    small = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else crop
    hs = cv2.resize(hole.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) if f < 1 else hole
    # FSR deja colores falsos (verdes) si el hueco toca el borde: se amplía
    # el recorte reflejando lo que hay alrededor
    p = 16
    small = cv2.copyMakeBorder(np.ascontiguousarray(small), p, p, p, p, cv2.BORDER_REFLECT)
    hs = cv2.copyMakeBorder(np.ascontiguousarray(hs.astype(np.uint8)), p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    known = np.where(hs > 0, 0, 255).astype(np.uint8)
    dst = np.zeros_like(small)
    cv2.xphoto.inpaint(small, known, dst, cv2.xphoto.INPAINT_FSR_FAST)
    dst = dst[p:-p, p:-p]
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
            # todo el daño del recorte cuenta como hueco (también las motas):
            # si no, el relleno copia la textura del desconchado de al lado
            hole = ((m[y0:y1, x0:x1] > 0) | face_hole[y0:y1, x0:x1]).astype(np.uint8)
            try:
                rec = fn(out[y0:y1, x0:x1], hole)
            except Exception:  # noqa: BLE001  si LaMa falla, relleno clásico
                rec = _fsr(out[y0:y1, x0:x1], hole)
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
