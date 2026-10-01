"""Restaurar cuadro: limpieza conservadora de pinturas.

Principio: devolver el cuadro a como se veía, no repintarlo. Por eso:
  - No hay reducción de ruido, ni enfoque, ni aumento de resolución: la
    pincelada, la textura de la tela y la materia de la pintura se quedan
    tal cual.
  - El barniz amarillento y la suciedad se corrigen con un ajuste global
    (igual en todo el cuadro), nunca con retoques locales que cambien la
    composición.
  - Las grietas del craquelado se cierran con interpolación desde sus bordes
    (son líneas de 1-3 px: no hay nada que inventar).
  - Las lagunas (pintura caída) se rellenan a partir de su entorno; la
    interfaz las propone en rosa para que las revises antes.
  - Las zonas descoloridas recuperan color a partir del color que las rodea,
    solo en los canales de color: la luz, y con ella la pincelada, no cambia.
  - Sobre los rasgos de una cara (ojos, nariz, boca) no se toca nada.
"""
from __future__ import annotations

import cv2
import numpy as np


def _lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


# --------------------------------------------------------------- barniz
def clean_varnish(rgb: np.ndarray, intensity: float) -> tuple[np.ndarray, dict]:
    """Barniz amarillento y suciedad: un velo cálido y apagado sobre todo el
    cuadro. Se estima con las zonas más claras (lo que el pintor pintó como
    blancos y luces tiende a neutro) y se corrige solo una parte, para no
    enfriar un cuadro que es cálido a propósito. También se recupera algo del
    contraste que quita la suciedad. Todo es global: la misma curva en todo
    el cuadro."""
    lab = _lab(rgb).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    lights = L >= np.percentile(L, 97)
    cast_a = float(np.median(a[lights]) - 128)
    cast_b = float(np.median(b[lights]) - 128)
    k = 0.35 + 0.45 * intensity  # fracción del tono amarillento que se quita
    da = -np.clip(cast_a, -12, 12) * k
    db = -np.clip(cast_b, -25, 25) * k
    # el velo pesa más en las luces que en las sombras
    wl = np.clip(L / 255.0, 0.25, 1.0)
    a2 = a + da * wl
    b2 = b + db * wl
    # suciedad: niveles (negro y blanco) con un recorte muy suave
    lo, hi = np.percentile(L, 0.5), np.percentile(L, 99.5)
    t = 0.3 + 0.5 * intensity
    lo_t, hi_t = lo * (1 - t), hi + (250 - hi) * t
    L2 = np.clip((L - lo) * (hi_t - lo_t) / max(1.0, hi - lo) + lo_t, 0, 255)
    out = cv2.merge([np.asarray(c, np.float32) for c in (L2, np.clip(a2, 0, 255), np.clip(b2, 0, 255))]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB), {"cast": (cast_a, cast_b), "removed": (da, db)}


# --------------------------------------------------------------- grietas
def detect_cracks(rgb: np.ndarray, intensity: float, feature_mask: np.ndarray | None = None) -> np.ndarray:
    """Craquelado: líneas oscuras finas (1-3 px) y alargadas. Los trazos que
    el pintor dibujó suelen ser más anchos o seguir bordes de color; aquí
    solo se aceptan líneas finas que contrastan con su entorno inmediato."""
    L = _lab(rgb)[..., 0]
    h, w = L.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k).astype(np.float32)
    mad = float(np.median(np.abs(bh - np.median(bh)))) + 1e-3
    thr = max(14.0, 6 * mad) * (1.25 - 0.5 * intensity)
    cand = (bh > thr).astype(np.uint8)
    # solo lo fino: lo que sobrevive a una apertura de 4x4 es una mancha o un trazo ancho
    wide = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    thin = cand & ~wide
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    out = np.zeros((h, w), np.uint8)
    min_len = 12
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if max(ww, hh) < min_len or area < 10:
            continue
        # alargada: poca área para su caja (una línea, no una mota)
        if area > 0.6 * ww * hh and min(ww, hh) > 4:
            continue
        out[y:y + hh, x:x + ww][lab[y:y + hh, x:x + ww] == i] = 255
    if feature_mask is not None:
        out[feature_mask > 0.5] = 0
    return cv2.dilate(out, np.ones((3, 3), np.uint8))


# --------------------------------------------------------------- lagunas
def detect_losses(rgb: np.ndarray, intensity: float, feature_mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Lagunas: pintura caída que deja ver la preparación (blanca o crema,
    casi sin color), mucho más clara que su entorno. Se usa la imagen sin
    limpiar: donde la pintura se ha caído también se fue el barniz, así que
    la laguna es más fría que los blancos pintados del cuadro (que están
    amarillentos). Un brillo pintado no se marca.
    Es una sugerencia: la interfaz la pinta en rosa para revisarla."""
    lab = _lab(rgb).astype(np.float32)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    h, w = L.shape
    bb = lab[..., 2] - 128
    lights = L >= np.percentile(L, 97)
    warm = float(np.median(bb[lights]))  # amarillo de los blancos del cuadro
    cand = ((L >= max(195.0, float(np.percentile(L, 96)))) & (chroma < 22)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_i, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros((h, w), np.uint8)
    regions = 0
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 20 or area > 0.01 * h * w:
            continue
        p = 10
        ys, xs = slice(max(0, y - p), min(h, y + hh + p)), slice(max(0, x - p), min(w, x + ww + p))
        comp = lab_i[ys, xs] == i
        ring = (cv2.dilate(comp.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~comp
        if not ring.any():
            continue
        l_in, l_ring = float(L[ys, xs][comp].mean()), float(np.median(L[ys, xs][ring]))
        if l_in - l_ring < 45 - 15 * intensity:
            continue
        if float(np.median(bb[ys, xs][comp])) > warm - (6 - 2 * intensity):
            continue  # tan amarillento como los blancos pintados: es pintura
        if feature_mask is not None and (comp & (feature_mask[ys, xs] > 0.5)).any():
            continue
        grown = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        out[ys, xs][grown] = 255
        regions += 1
    return out, regions


# --------------------------------------------------------------- color perdido
def revive_faded(rgb: np.ndarray, intensity: float, protect: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Zonas descoloridas: claramente más apagadas y más claras que lo que las
    rodea, sin tocar el borde del cuadro (un cielo gris o una pared blanca
    suelen llegar al borde y son así a propósito). En ellas se devuelve el
    color que aún conservan (mismo matiz) hasta acercarlo al de alrededor;
    donde el color se ha perdido del todo no se inventa. La luminosidad no
    cambia: la pincelada se conserva."""
    h, w = rgb.shape[:2]
    f = min(1.0, 384 / max(h, w))
    small = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    lab = _lab(small).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a - 128, b - 128)
    s = max(2.0, 0.012 * max(small.shape[:2]))
    Cb = cv2.GaussianBlur(C, (0, 0), s)
    Lb = cv2.GaussianBlur(L, (0, 0), s)
    big = int(0.12 * max(small.shape[:2])) | 1
    C_ref = cv2.dilate(Cb, np.ones((big, big), np.uint8))  # color de alrededor
    L_ref = cv2.blur(Lb, (big, big))
    faded = (Cb < 0.4 * C_ref) & (C_ref > 18) & (Lb > L_ref + 6 - 4 * intensity)
    faded = cv2.morphologyEx(faded.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab_i, stats, _ = cv2.connectedComponentsWithStats(faded, connectivity=8)
    keep = np.zeros_like(faded)
    sh, sw = faded.shape
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 0.002 * sh * sw or area > 0.15 * sh * sw:
            continue
        if x <= 1 or y <= 1 or x + ww >= sw - 1 or y + hh >= sh - 1:
            continue  # toca el borde: probablemente es así a propósito
        keep[lab_i == i] = 1
    if not keep.any():
        return rgb, 0.0
    # se aviva el color que aún queda (mismo matiz) hasta acercarlo al de
    # alrededor; donde no queda color no se inventa ninguno
    wgt = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), s * 1.5)
    wgt = np.clip(wgt * 1.4, 0, 1) * (0.45 + 0.45 * intensity)
    gain = np.clip(C_ref / (Cb + 2.0), 1.0, 2.5)
    k = 1.0 + (gain - 1.0) * wgt
    k = cv2.resize(k, (w, h), interpolation=cv2.INTER_CUBIC)
    if protect is not None:
        k = 1.0 + (k - 1.0) * (1 - np.clip(protect, 0, 1))
    full = _lab(rgb).astype(np.float32)
    full[..., 1] = np.clip(128 + (full[..., 1] - 128) * k, 0, 255)
    full[..., 2] = np.clip(128 + (full[..., 2] - 128) * k, 0, 255)
    out = cv2.cvtColor(full.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out, float(keep.mean())
