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


TEX_K = 6
WEAVE_K = 3.0
FLAKES_PER_MP = 350


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
    # los blancos menos amarillos del cuadro: en un cuadro cálido a propósito
    # siempre queda alguno neutro (y entonces apenas se corrige); bajo un
    # barniz amarillento hasta los más fríos amarillean
    cast_a = float(np.percentile(a[lights], 50) - 128)
    cast_b = float(np.percentile(b[lights], 10) - 128)
    k = 0.35 + 0.25 * intensity  # fracción del tono amarillento que se quita
    da = -np.clip(cast_a, -12, 12) * k
    db = -np.clip(cast_b, 0, 25) * k
    # el velo amarillo se nota sobre todo en las luces
    wl = np.clip(L / 255.0, 0.0, 1.0) ** 2
    a2 = a + da * wl
    b2 = b + db * wl
    # suciedad: niveles (negro y blanco) con un recorte muy suave
    lo, hi = np.percentile(L, 0.5), np.percentile(L, 99.5)
    t = 0.2 + 0.25 * intensity  # poco: no debe parecer recién pintado
    lo_t, hi_t = lo * (1 - t), hi + (250 - hi) * t
    L2 = np.clip((L - lo) * (hi_t - lo_t) / max(1.0, hi - lo) + lo_t, 0, 255)
    out = cv2.merge([np.asarray(c, np.float32) for c in (L2, np.clip(a2, 0, 255), np.clip(b2, 0, 255))]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB), {"cast": (cast_a, cast_b), "removed": (da, db)}


# --------------------------------------------------------------- grietas
NET = 0


STRAIGHT = 0.5
SOLID = 0.55  # zona clara maciza = objeto pintado, no racimo de desconchones
GROW = 3  # píxeles que se amplía cada desconchón hasta su borde real


def _straight(comp: np.ndarray) -> float:
    """Parte de la línea que es recta y larga. Las grietas zigzaguean; un
    trazo recto largo (jarcias, cables, marcos, contornos) es del pintor."""
    total = int(comp.sum())
    if total < 25:
        return 0.0
    segs = cv2.HoughLinesP(comp * 255, 1, np.pi / 90, 15, minLineLength=25, maxLineGap=2)
    if segs is None:
        return 0.0
    on = np.zeros_like(comp)
    for x0, y0, x1, y1 in segs[:, 0]:
        cv2.line(on, (int(x0), int(y0)), (int(x1), int(y1)), 1, 3)
    return float((on & comp).sum()) / total


def _cells(comp: np.ndarray) -> int:
    """Celdas cerradas que forma una red de líneas. El craquelado divide la
    pintura en «islas»; el pelo, las vetas o los trazos finos no se cierran."""
    cnt, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    return sum(1 for c, hc in zip(cnt, hier[0]) if hc[3] >= 0 and cv2.contourArea(c) >= 12)


def detect_cracks(rgb: np.ndarray, intensity: float, feature_mask: np.ndarray | None = None) -> np.ndarray:
    """Craquelado: líneas oscuras finas (1-3 px) y alargadas. Los trazos que
    el pintor dibujó suelen ser más anchos o seguir bordes de color; aquí
    solo se aceptan líneas finas que contrastan con su entorno inmediato."""
    L = _lab(rgb)[..., 0]
    h, w = L.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k).astype(np.float32)
    # textura de la pintura sin las líneas finas (la mediana las borra): una
    # grieta solo se reconoce si destaca mucho sobre la textura de alrededor.
    # Pelo, encajes, vetas o pinceladas finas forman parte de la obra
    base = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k).astype(np.float32)
    mu = cv2.blur(base, (15, 15))
    tex = np.sqrt(np.maximum(cv2.blur(base * base, (15, 15)) - mu * mu, 0))
    # la trama del lienzo deja surcos finos por todo el cuadro: una grieta
    # tiene que ser bastante más oscura que ese fondo de trama
    weave = cv2.blur(bh, (21, 21))
    thr = np.maximum(14.0, WEAVE_K * weave) * (1.25 - 0.5 * intensity)
    cand = ((bh > thr) & (tex < TEX_K)).astype(np.uint8)
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
        comp = (lab[y:y + hh, x:x + ww] == i).astype(np.uint8)
        if NET and _cells(comp) < NET:
            continue
        if STRAIGHT and _straight(comp) > STRAIGHT:
            continue  # trazo recto (jarcias, cables, contornos): lo pintó alguien
        out[y:y + hh, x:x + ww][comp > 0] = 255
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
        if float(np.mean(L[ys, xs][comp] >= 250)) > 0.5:
            continue  # blanco puro, saturado: un reflejo pintado, no la preparación
        if _halo(L[ys, xs], comp) > HALO:
            continue  # brillo pintado: se apaga poco a poco hacia fuera
        if feature_mask is not None:
            # sobre un rasgo solo cuenta si casi toda la laguna está fuera de
            # él (una rotura que cruza la cara); un brillo del ojo, no
            inside = (comp & (feature_mask[ys, xs] > 0.5)).sum()
            if inside and inside > 0.4 * area:
                continue
        grown = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        out[ys, xs][grown] = 255
        regions += 1
    strips, n_strips = _strips(L, chroma, intensity, feature_mask)
    return np.maximum(out, strips), regions + n_strips


HALO = 12.0


def _halo(L, comp):
    """Cuánto más claro es el borde inmediato que lo de un poco más allá. Un
    brillo pintado (reflejo de una cuchara, de un ojo) se va apagando hacia
    fuera; una laguna corta en seco con la pintura de alrededor."""
    c = comp.astype(np.uint8)
    d3 = cv2.dilate(c, np.ones((5, 5), np.uint8)) > 0
    d7 = cv2.dilate(c, np.ones((9, 9), np.uint8)) > 0
    d13 = cv2.dilate(c, np.ones((15, 15), np.uint8)) > 0
    inner, outer = d3 & ~comp, d13 & ~d7
    if not inner.any() or not outer.any():
        return 0.0
    return float(np.median(L[inner]) - np.median(L[outer]))


def _strips(L, chroma, intensity, feature_mask):
    """Roturas y pintura levantada en franja: una banda larga y estrecha,
    clara y sin color, con el borde oscuro (la sombra del levantamiento).
    Aquí no hace falta que sea más fría que los blancos: la forma la delata."""
    h, w = L.shape
    cand = ((L >= 185) & (chroma < 22)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_i, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros((h, w), np.uint8)
    count = 0
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 0.002 * h * w or area > 0.08 * h * w:
            continue
        p = 10
        ys, xs = slice(max(0, y - p), min(h, y + hh + p)), slice(max(0, x - p), min(w, x + ww + p))
        comp = lab_i[ys, xs] == i
        thick = 2 * float(cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5).max())
        length = area / max(1.0, thick)
        if thick > 0.03 * max(h, w) + 4 or length < 12 * thick:
            continue
        ring = (cv2.dilate(comp.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~comp
        if float(L[ys, xs][comp].mean()) - float(np.median(L[ys, xs][ring])) < 60 - 20 * intensity:
            continue
        if feature_mask is not None:
            inside = (comp & (feature_mask[ys, xs] > 0.5)).sum()
            if inside > 0.4 * area:
                continue
        grown = cv2.dilate(comp.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # con su sombra
        out[ys, xs][grown] = 255
        count += 1
    return out, count


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
        comp = lab_i == i
        # solo si es el MISMO color, apagado: un gato blanco o un cuello
        # blanco sobre un vestido rojo no son rojo descolorido
        ring = (cv2.dilate(comp.astype(np.uint8), np.ones((big, big), np.uint8)) > 0) & ~comp
        ca, cb = a[comp].mean() - 128, b[comp].mean() - 128
        ra, rb = a[ring].mean() - 128, b[ring].mean() - 128
        if np.hypot(ca, cb) < 6:
            continue  # casi sin color: no hay matiz que avivar
        cosang = (ca * ra + cb * rb) / (np.hypot(ca, cb) * np.hypot(ra, rb) + 1e-6)
        if cosang < np.cos(np.radians(25)):
            continue
        keep[comp] = 1
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


# --------------------------------------------------------------- pincel
def _local_stats(lab: np.ndarray, valid: np.ndarray, sigma: float):
    """Color medio y dispersión de la pintura sana alrededor de cada punto."""
    v = valid.astype(np.float32)
    wv = cv2.GaussianBlur(v, (0, 0), sigma) + 1e-4
    mean = np.stack([cv2.GaussianBlur(lab[..., c] * v, (0, 0), sigma) for c in range(3)], -1) / wv[..., None]
    sq = np.stack([cv2.GaussianBlur(lab[..., c] ** 2 * v, (0, 0), sigma) for c in range(3)], -1) / wv[..., None]
    std = np.sqrt(np.maximum(sq - mean ** 2, 0).sum(-1))
    return mean, std, wv


def refine_brush(rgb: np.ndarray, brush: np.ndarray, intensity: float) -> np.ndarray:
    """De lo pintado con el pincel, solo lo que de verdad está dañado: la
    preparación o el lienzo que asoma y los bordes de la rotura, que se
    apartan claramente del color de la pintura sana de alrededor. La pintura
    buena que el pincel haya cubierto no se toca."""
    b = brush > 0
    if not b.any():
        return brush
    lab = _lab(rgb).astype(np.float32)
    # grosor típico del trazo del pincel → escala a la que mirar alrededor
    dist = cv2.distanceTransform(b.astype(np.uint8), cv2.DIST_L2, 5)
    sigma = max(3.0, 1.0 * float(np.percentile(dist[b], 90)))
    valid = ~b
    dmg = b.copy()
    h, w = b.shape
    # un píxel del pincel está sano si alrededor hay pintura sana de su mismo
    # color (cualquiera de los colores de alrededor, no la media: el pincel
    # puede cruzar el borde entre la cara y la ropa)
    for radii in ((sigma + 1, 1.4 * sigma + 2, 1.8 * sigma + 3),):
        best = np.full((h, w), np.inf, np.float32)
        for r in radii:
            for t in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                dx, dy = int(round(r * np.cos(t))), int(round(r * np.sin(t)))
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                sh = cv2.warpAffine(lab, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
                sv = cv2.warpAffine(valid.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
                d = np.sqrt(((lab - sh) ** 2).sum(-1))
                d[~sv] = np.inf
                np.minimum(best, d, out=best)
        thr = 16.0 * (1.2 - 0.4 * intensity)
        dmg = b & (best > thr)
        valid = ~b | ~cv2.dilate(dmg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # lo que tiene el mismo color que el daño encontrado y lo toca también es
    # daño (p. ej. la rotura al pasar junto a un cuello blanco)
    if dmg.sum() >= 20:
        k = int(min(3, dmg.sum() // 20))
        pts = lab[dmg].astype(np.float32)
        _, _, centers = cv2.kmeans(pts, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                   2, cv2.KMEANS_PP_CENTERS)
        near = np.min(np.stack([np.sqrt(((lab - c) ** 2).sum(-1)) for c in centers]), 0)
        like = (b & (near < 10.0 * (1.2 - 0.4 * intensity))) | dmg
        grown = dmg.astype(np.uint8)
        for _ in range(64):
            nxt = cv2.dilate(grown, np.ones((3, 3), np.uint8)) & like.astype(np.uint8)
            if (nxt == grown).all():
                break
            grown = nxt
        dmg = grown > 0
    m = cv2.morphologyEx(dmg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8)) & b.astype(np.uint8)
    if m.sum() < 0.02 * b.sum():
        return brush  # nada destaca: se respeta lo que marcaste
    return (m > 0).astype(np.uint8) * 255


# --------------------------------------------------------------- desconchados
def _painted_whites(Lf: np.ndarray, C: np.ndarray, size: int) -> np.ndarray:
    """Objetos pintados de blanco o crema (un gato blanco, un cuello): zonas
    claras grandes y macizas. Su borde contra la pintura de alrededor no es
    un desconchón. Una zona desconchada, en cambio, es un racimo de islas
    sueltas: grande pero hueca. Devuelve los objetos macizos con su borde: dentro, una
    laguna blanca sobre blanco no se distingue de la pintura y se deja."""
    light = ((Lf > 185) & (C < 34)).astype(np.uint8)
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_i, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=8)
    out = np.zeros(Lf.shape, bool)
    min_area = (0.04 * size) ** 2
    band = max(3, int(0.006 * size)) | 1
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = stats[j, :4]
        comp = (lab_i[y:y + h, x:x + w] == j).astype(np.uint8)
        filled = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        hull = cv2.convexHull(cv2.findNonZero(filled))
        if comp.sum() / max(1.0, cv2.contourArea(hull)) < SOLID:
            continue
        out[y:y + h, x:x + w] |= cv2.dilate(filled, np.ones((band, band), np.uint8)) > 0
    return out


def detect_flakes(rgb: np.ndarray, intensity: float, feature_mask: np.ndarray | None = None) -> np.ndarray:
    """Desconchados: muchas islas de pintura caída que dejan ver la
    preparación (crema o blanca, casi sin color), claramente más claras que
    la pintura que las rodea. Primero se buscan las evidentes; con ellas se
    aprende el color de la preparación de ESTE cuadro, y luego se buscan las
    que caen sobre zonas claras (cara, brazos), donde se distinguen por ese
    color y no tanto por la luz. Sobre ojos, nariz y boca solo cuenta lo que
    tiene exactamente ese color."""
    lab = _lab(rgb)
    L = lab[..., 0]
    Lf = L.astype(np.float32)
    a = lab[..., 1].astype(np.float32) - 128
    b = lab[..., 2].astype(np.float32) - 128
    C = np.hypot(a, b)
    h, w = L.shape
    k = max(15, int(0.03 * max(h, w))) | 1
    bg = cv2.medianBlur(L, min(k, 255)).astype(np.float32)
    s = 1.0 - 0.4 * intensity
    strong = (Lf - bg > 30 * s) & (C < 32) & (Lf > 170)
    if feature_mask is not None:
        strong &= feature_mask < 0.5
    strong = cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    # solo si el cuadro está de verdad desconchado: cientos de islas, y del
    # color crema de una preparación (no los brillos blancos de la pintura)
    n_isl = cv2.connectedComponents(strong.astype(np.uint8), connectivity=8)[0] - 1
    if (n_isl < FLAKES_PER_MP * h * w / 1e6 or float(np.mean(C[strong] >= 8)) < 0.5
            or float(np.mean(L[strong] >= 248)) > 0.25):
        return np.zeros((h, w), np.uint8)
    # color de la preparación y fondo sin ella (media de lo que no es desconchado)
    g = np.array([np.median(Lf[strong]), np.median(a[strong]), np.median(b[strong])], np.float32)
    keep = (~cv2.dilate(strong.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).astype(np.float32)
    sg = max(4.0, 0.012 * max(h, w))
    bg2 = cv2.GaussianBlur(Lf * keep, (0, 0), sg) / (cv2.GaussianBlur(keep, (0, 0), sg) + 1e-3)
    dist = np.sqrt((Lf - g[0]) ** 2 * 0.5 + (a - g[1]) ** 2 + (b - g[2]) ** 2)
    by_color = (dist < 14 * (0.8 + 0.4 * intensity)) & (Lf - bg2 > 12 * s)
    # los desconchados van en racimos: lejos de los evidentes no se busca
    r = max(9, int(0.02 * max(h, w))) | 1
    by_color &= cv2.dilate(strong.astype(np.uint8), np.ones((r, r), np.uint8)) > 0
    if feature_mask is not None:
        on_feat = feature_mask >= 0.5
        by_color &= ~on_feat | ((dist < 11) & (Lf - bg2 > 20 * s))
    # un blanco puro y saturado es un reflejo pintado, no la preparación
    strong &= Lf < 248
    whites = _painted_whites(Lf, C, max(h, w))
    m = (strong | by_color) & ~whites
    # el borde de cada desconchón: la preparación sigue unos píxeles más
    # allá de lo que salta a la vista (si no, queda un cerco claro)
    ring = (dist < 18 * (0.8 + 0.4 * intensity)) & (Lf - bg2 > 4 * s) & ~whites
    if feature_mask is not None:
        ring &= feature_mask < 0.5
    for _ in range(GROW):
        m |= (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ring
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # trocitos sueltos de 1-2 px son grano o brillo, no desconchados
    n, lab_i, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < 6
    small[0] = False
    m[small[lab_i]] = 0
    # racimo denso (la pintura caída a trozos, como un colador): se rellena
    # entero, porque lo que queda entre los huecos también es preparación
    kd = max(5, int(0.009 * max(h, w))) | 1
    dense = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))) > 0
    dense &= ~whites
    if feature_mask is not None:
        dense &= feature_mask < 0.5
    m |= dense.astype(np.uint8)
    return cv2.dilate(m, np.ones((3, 3), np.uint8)) * 255
