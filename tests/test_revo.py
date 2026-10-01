"""Pruebas de fidelidad de ONFR REVO.   python tests/test_revo.py

1. Una foto limpia apenas se toca en modo Restaurar (como mucho algún brillo
   pequeño que el detector heurístico confunde con polvo).
2. En una foto antigua sintética se reparan defectos, pero nunca sobre rasgos.
3. La cara procesada pasa la verificación de identidad, rasgos y estructura.
4. El verificador DETECTA una cara alterada (expresión cambiada, como haría
   un modelo generativo que "mejora" de más) y la rechaza.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from revo import Settings, process  # noqa: E402
from revo.faces import MOUTH, EYE_L, EYE_R, FaceGuard, face_masks  # noqa: E402
from revo.fidelity import check_face, identity_limit  # noqa: E402
from revo.restore import detect_defects  # noqa: E402
from revo.analysis import estimate_noise  # noqa: E402


def load(name):
    path = os.path.join(ROOT, "samples", name)
    if not os.path.exists(path):
        import subprocess
        subprocess.check_call([sys.executable, os.path.join(ROOT, "tests", "make_samples.py")])
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def test_clean_photo_barely_touched():
    rgb = load("astronauta_original.png")
    r = process(rgb, Settings(mode="restaurar", intensity=0.35))
    frac = r.stats["defect_pixels"] / (rgb.shape[0] * rgb.shape[1])
    assert frac < 0.003, f"demasiados 'defectos' en una foto limpia: {frac:.4%}"
    assert r.faces and r.faces[0].check.passed


def test_old_photo_defects_never_on_features():
    rgb = load("foto_antigua.png")
    g = FaceGuard.get()
    faces = g.detect(rgb)
    assert faces, "no se detectó la cara"
    face_m, feat_m = face_masks(rgb.shape, faces[0])
    for inten in (0.2, 0.5, 0.9):
        d = detect_defects(rgb, estimate_noise(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)), face_m, feat_m, inten)
        assert d.any(), "no se detectó ningún defecto en la foto antigua"
        assert not (d[feat_m > 0.5] > 0).any(), "se marcó un defecto sobre ojos/nariz/boca"


def test_processed_face_passes_verification():
    for mode in ("restaurar", "mejorar_restaurar"):
        for inten in (0.2, 0.8):
            r = process(load("foto_antigua.png"), Settings(mode=mode, intensity=inten))
            assert r.faces, "sin caras"
            for f in r.faces:
                assert f.check.passed or f.applied == 0, f.check


def warp_mouth_and_eyes(rgb, face):
    """Simula una 'mejora' generativa que altera la expresión: sonrisa más
    ancha y ojos más grandes (lo que REVO nunca debe aceptar)."""
    h, w = rgb.shape[:2]
    mapx, mapy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    io = face.interocular
    for group, amount in ((MOUTH, 0.22), (EYE_L, 0.35), (EYE_R, 0.35)):
        c = face.landmarks[group].mean(0)
        dx, dy = mapx - c[0], mapy - c[1]
        r2 = (dx * dx + dy * dy) / (0.5 * io) ** 2
        f = amount * np.exp(-r2)
        mapx -= dx * f
        mapy -= dy * f
    return cv2.remap(rgb, mapx, mapy, cv2.INTER_LINEAR)


def test_guard_rejects_altered_face():
    rgb = load("astronauta_original.png")
    g = FaceGuard.get()
    face = g.detect(rgb)[0]
    face_m, _ = face_masks(rgb.shape, face)
    lim = identity_limit(g, face, rgb, 1.0)
    same = check_face(g, face, rgb, rgb, face_m, id_limit=lim)
    assert same.passed, same
    altered = warp_mouth_and_eyes(rgb, face)
    chk = check_face(g, face, altered, rgb, face_m, id_limit=lim)
    print("   cara alterada →", chk)
    assert not chk.passed, "el verificador no detectó una cara alterada"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("OK  ", name)
            except AssertionError as e:
                fails += 1
                print("FALLO", name, e)
    sys.exit(1 if fails else 0)
