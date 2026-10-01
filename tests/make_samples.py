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


def save(name, rgb):
    cv2.imwrite(os.path.join(OUT, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("->", name, rgb.shape)


astro = data.astronaut()
save("astronauta_original.png", astro)
save("foto_antigua.png", old_photo(astro))
save("foto_rota.png", torn_photo(astro))
save("foto_baja_calidad.jpg", low_quality(astro))
save("cafe_baja_calidad.jpg", low_quality(data.coffee()))
