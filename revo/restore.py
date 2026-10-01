"""Restauración conservadora: polvo, arañazos, ruido y contraste.

Reglas que sigue este módulo:
  - Solo se rellenan defectos pequeños y claramente identificables.
  - Nunca se rellena nada sobre ojos, cejas, nariz o boca.
  - Sobre la piel de una cara solo se quitan motas claras diminutas (polvo
    típico de escaneo); las manchas oscuras pueden ser lunares o pecas y se
    respetan.
  - Se conserva parte del grano original para que una foto antigua siga
    pareciendo de su época.
"""
from __future__ import annotations

import cv2
import numpy as np


def _odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 else n + 1


def detect_defects(
    rgb: np.ndarray,
    noise: float,
    face_mask: np.ndarray | None = None,
    feature_mask: np.ndarray | None = None,
    intensity: float = 0.5,
    report: dict | None = None,
) -> np.ndarray:
    """Máscara uint8 (255 = defecto a reparar). Si se pasa `report`, anota
    cuántos defectos se dejaron sin tocar por estar sobre una cara."""
    skipped_face = 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    k = _odd(max(5, min(h, w) / 160))
    # top-hat / black-hat: solo responden a estructuras claras u oscuras más
    # estrechas que k (motas, líneas finas); un borde normal da cero
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se).astype(np.int16)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se).astype(np.int16)
    diff = np.where(tophat >= blackhat, tophat, -blackhat)

    # cuanto más conservador, más contraste se exige para considerar algo un defecto
    t = max(35.0, noise * 5.5) * (1.25 - 0.5 * intensity)
    bright = (diff > t).astype(np.uint8)
    dark = (diff < -t).astype(np.uint8)

    # textura del entorno de cada píxel, medida sin los candidatos (que se
    # sustituyen por la mediana): un defecto es una anomalía sobre zona lisa;
    # si alrededor hay textura, puede ser detalle real y se respeta
    cand = cv2.dilate(bright | dark, np.ones((3, 3), np.uint8)) > 0
    clean = np.where(cand, cv2.medianBlur(gray, k), gray).astype(np.float32)
    mu = cv2.blur(clean, (5, 5))
    local_std = np.sqrt(np.maximum(cv2.blur(clean * clean, (5, 5)) - mu * mu, 0))
    texture = cv2.blur(local_std, (2 * k + 1, 2 * k + 1))
    # el polvo es disperso: donde casi todo es "candidato" (pelo, tejidos,
    # hierba) se trata de textura real
    density = cv2.blur(cand.astype(np.float32), (4 * k + 1, 4 * k + 1))
    smooth_enough = (np.abs(diff) > np.maximum(t, 4.0 * texture)) & (density < 0.3)

    max_speck = (k * k) * (0.6 + 0.8 * intensity)
    min_scratch_len = max(16, min(h, w) / 20)
    out = np.zeros_like(gray)

    for cand, is_bright in ((bright, True), (dark, False)):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            region = (slice(y, y + hh), slice(x, x + ww))
            comp = lab[region] == i
            is_speck = area <= max_speck
            is_scratch = False
            if not is_speck:
                ys, xs = np.nonzero(comp)
                pts = np.column_stack([xs, ys]).astype(np.float32)
                (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
                length, thick = max(rw, rh), max(1.0, min(rw, rh))
                # arañazo: largo, fino y recto (área ≈ largo × grosor)
                is_scratch = length >= min_scratch_len and thick <= 4 + 0.02 * length and area <= 5 * length
            if not (is_speck or is_scratch):
                continue
            ok = smooth_enough[region] & comp
            if is_speck:
                if ok.sum() < 0.7 * area:
                    continue
                comp = comp.copy()
            else:
                # en un arañazo largo se reparan solo los tramos sobre zona lisa
                comp = ok
                if comp.sum() < 0.3 * area:
                    continue
            on_feature = feature_mask is not None and (feature_mask[region][comp] > 0.5).any()
            on_face = face_mask is not None and (face_mask[region][comp] > 0.05).any()
            if on_face or on_feature:
                if is_scratch:
                    # arañazo fino sobre la piel: se repara solo fuera de los rasgos
                    if feature_mask is not None:
                        comp &= feature_mask[region] <= 0.5
                    skipped_face += int(on_feature)
                elif on_feature or not (is_bright and area <= max(6, max_speck * 0.7)):
                    # jamás sobre rasgos; en piel solo motas claras diminutas
                    # (una mancha oscura puede ser un lunar o una peca)
                    skipped_face += 1
                    continue
            out[region][comp] = 255
    if report is not None:
        report["skipped_on_faces"] = skipped_face
    return out


def repair_defects(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return rgb.copy()
    m = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return cv2.inpaint(rgb, m, 3, cv2.INPAINT_TELEA)


def denoise(rgb: np.ndarray, noise: float, intensity: float, monochrome: bool) -> np.ndarray:
    if noise < 1.5:
        return rgb.copy()
    h = float(np.clip(noise * (0.7 + 0.7 * intensity), 2.0, 16.0))
    if monochrome:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.fastNlMeansDenoising(lab[..., 0], None, h, 7, 21)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return cv2.fastNlMeansDenoisingColored(rgb, None, h, h * 0.8, 7, 21)


def keep_grain(original: np.ndarray, denoised: np.ndarray, amount: float) -> np.ndarray:
    """Devuelve parte del grano eliminado (carácter de época)."""
    if amount <= 0:
        return denoised
    o, d = original.astype(np.float32), denoised.astype(np.float32)
    return np.clip(d + amount * (o - d), 0, 255).astype(np.uint8)


def tone_lut(rgb: np.ndarray, intensity: float) -> np.ndarray:
    """LUT global (punto a punto) para la luminancia: niveles y gamma moderados.
    Al ser global, se puede aplicar igual a la versión mínima de las caras."""
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    lo, hi = np.percentile(L, [0.4, 99.6])
    s = 0.35 + 0.5 * intensity  # cuánto del estiramiento completo se aplica
    lo_t, hi_t = lo * (1 - s), hi + (255 - hi) * s
    x = np.arange(256, dtype=np.float32)
    y = (x - lo) / max(1.0, hi - lo) * (hi_t - lo_t) + lo_t
    # gamma suave hacia un gris medio sin forzar
    mean = float(np.clip(np.interp(L.mean(), x, y), 1, 254)) / 255.0
    g = np.log(0.5) / np.log(mean)
    g = 1.0 + (g - 1.0) * 0.35 * intensity
    g = float(np.clip(g, 0.8, 1.25))
    y = 255.0 * np.power(np.clip(y, 0, 255) / 255.0, g)
    return np.clip(y, 0, 255).astype(np.uint8)


def apply_tone(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def local_contrast(rgb: np.ndarray, intensity: float) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.2 + 1.3 * intensity, tileGridSize=(8, 8))
    L = lab[..., 0]
    alpha = 0.2 + 0.35 * intensity
    lab[..., 0] = cv2.addWeighted(clahe.apply(L), alpha, L, 1 - alpha, 0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def detect_damage(
    rgb: np.ndarray,
    monochrome: bool,
    face_mask: np.ndarray | None = None,
    feature_mask: np.ndarray | None = None,
    intensity: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Daños grandes que `detect_defects` no cubre: roturas, papel arrancado,
    marcas o desconchones. Solo en fotos en blanco y negro o viradas, donde
    el daño se delata porque su color no sigue el tono de la foto (el papel
    blanco o el soporte que asoma). En fotos en color se usa el pincel.

    Nunca marca nada sobre una cara ni pegado al borde de la foto (los bordes
    rotos se reparan con el pincel).
    """
    from .analysis import tone_deviation

    h, w = rgb.shape[:2]
    out = np.zeros((h, w), np.uint8)
    info = {"regions": 0}
    if not monochrome:
        return out, info
    dev = tone_deviation(rgb)
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    thr = 10.0 - 3.0 * intensity
    cand = (dev > thr).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    min_area = max(30, h * w // 20000)
    margin = int(0.02 * min(h, w))
    face = (face_mask > 0.05) if face_mask is not None else np.zeros((h, w), bool)
    feat = (feature_mask > 0.5) if feature_mask is not None else np.zeros((h, w), bool)
    soft = (dev > 0.6 * thr).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 25 or area > 0.015 * h * w:
            continue
        if x < margin or y < margin or x + ww > w - margin or y + hh > h - margin:
            continue
        # todo se calcula sobre un recorte alrededor del daño (antes, sobre la
        # imagen entera por cada componente: unos 2 s en total)
        p = max(4, int(np.sqrt(area) * 0.3)) + 12
        ys = slice(max(0, y - p), min(h, y + hh + p))
        xs = slice(max(0, x - p), min(w, x + ww + p))
        comp = lab[ys, xs] == i
        Lc, fc, ftc = L[ys, xs], face[ys, xs], feat[ys, xs]
        ring = (cv2.dilate(comp.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~comp
        d_mean = float(dev[ys, xs][comp].mean())
        l_in, l_ring = float(Lc[comp].mean()), float(np.median(Lc[ring]))
        # rotura grande: zona amplia claramente fuera de tono
        # (si solo cambia el color pero no la luminosidad, es una mancha: se
        # corrige con retone_stains sin rellenar nada)
        big_tear = area >= 0.001 * h * w and d_mean >= 11 - 2 * intensity and abs(l_in - l_ring) >= 25
        # desconchón o marca: papel claro sobre zona bastante más oscura
        bright_mark = d_mean >= 9 and l_in >= 160 and l_in - l_ring >= 45 - 10 * intensity
        if area < min_area and not (bright_mark and l_in - l_ring >= 60):
            continue
        if not (big_tear or bright_mark):
            continue
        if (comp & (fc | ftc)).any():
            continue
        # crecer hacia el borde de papel roto que rodea el daño: blanco (más
        # claro que el fondo) o fuera de tono
        r = max(4, int(np.sqrt(area) * (0.3 if big_tear else 0.15)))
        grown = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        paper = Lc >= max(222.0, l_ring + 12)
        edge = grown & ((soft[ys, xs] > 0) | paper)
        edge = cv2.morphologyEx(edge.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        region = (comp | edge) & ~fc & ~ftc
        region = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        out[ys, xs][region & ~ftc] = 255
        info["regions"] += 1
    return out, info


def retone_stains(rgb: np.ndarray, intensity: float, protect: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Manchas de color en una foto en blanco y negro o virada (humedad,
    químicos): se devuelve su color al tono de la foto. Solo cambia el color;
    la luminosidad, y con ella todo el detalle, queda intacta."""
    from .analysis import tone_deviation

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    bins = np.clip((L / 16).astype(np.int32), 0, 15)
    ea = np.full(16, np.median(a), np.float32)
    eb = np.full(16, np.median(b), np.float32)
    for i in range(16):
        sel = bins == i
        if sel.sum() > 50:
            ea[i], eb[i] = np.median(a[sel]), np.median(b[sel])
    dev = tone_deviation(rgb)
    thr = 9.0 - 3.0 * intensity
    wgt = np.clip((dev - thr) / thr, 0, 1) * (0.6 + 0.4 * intensity)
    wgt = cv2.GaussianBlur(wgt, (0, 0), 3)
    if protect is not None:
        wgt *= 1 - np.clip(protect, 0, 1)
    lab[..., 1] = a + wgt * (ea[bins] - a)
    lab[..., 2] = b + wgt * (eb[bins] - b)
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out, float((wgt > 0.2).mean())
