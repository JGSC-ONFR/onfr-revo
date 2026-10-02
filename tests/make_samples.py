"""Genera imágenes de prueba degradadas a partir de fotos de dominio público
incluidas en scikit-image (astronauta: NASA, dominio público)."""
import os

import cv2
import numpy as np
from skimage import data

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1930)


def old_photo(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    g = 60 + g * 0.6  # contraste perdido
    g += rng.normal(0, 9, g.shape)  # grano
    h, w = g.shape
    for _ in range(140):  # polvo claro
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(g, (int(x), int(y)), int(rng.integers(1, 3)), 235, -1)
    for _ in range(50):  # motas oscuras
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(g, (int(x), int(y)), 1, 25, -1)
    for _ in range(4):  # arañazos
        x = int(rng.integers(0, w))
        cv2.line(g, (x, 0), (x + int(rng.integers(-30, 30)), h - 1), 230, 1)
    g = np.clip(g, 0, 255)
    sepia = np.stack([g * 1.0, g * 0.88, g * 0.70], -1)
    return np.clip(sepia, 0, 255).astype(np.uint8)


def torn_photo(rgb):
    """Foto antigua con una rotura junto a la cara (se ve el soporte rojizo
    con borde de papel blanco) y marcas blancas gruesas sobre ropa oscura."""
    img = old_photo(rgb)
    h, w = img.shape[:2]
    pts = np.array([[300, 70], [326, 58], [345, 80], [338, 120], [312, 128], [298, 100]], np.int32)
    paper = np.zeros((h, w), np.uint8)
    cv2.fillPoly(paper, [pts], 255)
    paper = cv2.dilate(paper, np.ones((7, 7), np.uint8))
    img[paper > 0] = (238, 236, 230)
    inner = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inner, [pts], 255)
    inner = cv2.erode(inner, np.ones((3, 3), np.uint8))
    img[inner > 0] = (150, 70, 55) + rng.normal(0, 6, (int((inner > 0).sum()), 3))
    for (x, y, ww, hh) in ((185, 205, 22, 9), (215, 215, 8, 18), (240, 200, 14, 7)):
        cv2.ellipse(img, (x, y), (ww // 2, hh // 2), 20, 0, 360, (225, 222, 215), -1)
    return img


def low_quality(rgb):
    small = cv2.resize(rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    small = np.clip(small + rng.normal(0, 6, small.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 30])
    return cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)


def painting(rgb):
    """Imitación de óleo: pinceladas (oilPainting) sobre la foto ampliada."""
    big = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    bgr = cv2.xphoto.oilPainting(cv2.cvtColor(big, cv2.COLOR_RGB2BGR), 7, 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crack_mask(h, w, n=90):
    """Craquelado: red de grietas finas que se ramifican."""
    m = np.zeros((h, w), np.uint8)
    for _ in range(n):
        x, y = float(rng.uniform(0, w)), float(rng.uniform(0, h))
        ang = float(rng.uniform(0, 2 * np.pi))
        for _ in range(int(rng.integers(6, 18))):
            ang += float(rng.normal(0, 0.5))
            step = float(rng.uniform(6, 16))
            nx, ny = x + step * np.cos(ang), y + step * np.sin(ang)
            cv2.line(m, (int(x), int(y)), (int(nx), int(ny)), 255, int(rng.integers(1, 3)))
            x, y = nx, ny
    return m


def damaged_painting(rgb):
    """Barniz amarillento y sucio, craquelado, una zona descolorida y
    lagunas (pintura caída que deja ver la preparación blanca)."""
    img = rgb.astype(np.float32)
    h, w = img.shape[:2]
    # zona descolorida (pérdida de color, más clara)
    fade = np.zeros((h, w), np.float32)
    cv2.ellipse(fade, (int(w * 0.72), int(h * 0.62)), (int(w * 0.12), int(h * 0.09)), 20, 0, 360, 1, -1)
    fade = cv2.GaussianBlur(fade, (0, 0), 12)[..., None]
    gray = img.mean(-1, keepdims=True)
    img = img * (1 - 0.8 * fade) + (gray * 0.85 + 45) * 0.8 * fade
    # barniz amarillento + suciedad (menos contraste)
    img = 30 + img * 0.78
    img *= np.array([1.0, 0.92, 0.72], np.float32)
    # craquelado
    cm = crack_mask(h, w) > 0
    img[cm] *= 0.45
    # lagunas
    for _ in range(4):
        cx, cy = int(rng.uniform(0.1, 0.9) * w), int(rng.uniform(0.1, 0.9) * h)
        pts = np.array([[cx + rng.integers(-18, 18), cy + rng.integers(-18, 18)] for _ in range(6)], np.int32)
        cv2.fillPoly(img, [cv2.convexHull(pts)], (236, 230, 214))
    return np.clip(img, 0, 255).astype(np.uint8), cm


def torn_portrait(rgb):
    """Retrato pequeño (300 px) con la pintura levantada en una franja que
    cruza ojo, nariz, boca y ropa, y el trazo de pincel (generoso) con que
    se marcaría."""
    r = np.random.default_rng(1)
    oil = cv2.resize(painting(rgb[0:400, 40:440]), (300, 300), interpolation=cv2.INTER_AREA)
    pts = [(95, 60), (120, 95), (135, 118), (148, 140), (160, 160), (170, 185), (190, 230), (215, 295)]
    tear = np.zeros((300, 300), np.uint8)
    for a, b in zip(pts, pts[1:]):
        cv2.line(tear, a, b, 255, int(r.integers(5, 9)))
    tear = cv2.morphologyEx(tear, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    shadow = cv2.dilate(tear, np.ones((5, 5), np.uint8)) & ~tear
    img = oil.astype(np.float32)
    img[shadow > 0] *= 0.55
    ground = np.array([226, 232, 238], np.float32) + r.normal(0, 4, (300, 300, 1))
    img[tear > 0] = ground[tear > 0]
    brush = cv2.dilate(tear, np.ones((15, 15), np.uint8))
    return oil, np.clip(img, 0, 255).astype(np.uint8), tear, brush


def save(name, rgb):
    cv2.imwrite(os.path.join(OUT, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("->", name, rgb.shape)


astro = data.astronaut()
save("astronauta_original.png", astro)
save("foto_antigua.png", old_photo(astro))
save("foto_rota.png", torn_photo(astro))
save("foto_baja_calidad.jpg", low_quality(astro))
save("cafe_baja_calidad.jpg", low_quality(data.coffee()))

cuadro = painting(astro)
save("cuadro_original.png", cuadro)
save("cuadro_deteriorado.png", damaged_painting(cuadro)[0])
bodegon = painting(data.coffee())
save("bodegon_original.png", bodegon)
save("bodegon_deteriorado.png", damaged_painting(bodegon)[0])
oil, torn, tear, brush = torn_portrait(astro)
save("retrato_original.png", oil)
save("retrato_roto.png", torn)
cv2.imwrite(os.path.join(OUT, "retrato_rotura.png"), tear)
cv2.imwrite(os.path.join(OUT, "retrato_pincel.png"), brush)
