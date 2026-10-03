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


def test_torn_photo_hole_is_filled_but_face_untouched():
    rgb = load("foto_rota.png")
    r = process(rgb, Settings(mode="restaurar", intensity=0.35))
    assert r.stats["damage_regions"] >= 1, "no se detectó la rotura"
    hole = r._masks["damage"] > 0
    lab = cv2.cvtColor(r.image, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    before = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma0 = np.hypot(before[..., 1] - 128, before[..., 2] - 128)
    assert chroma[hole].mean() < chroma0[hole].mean() - 10, "la rotura sigue mostrando el soporte rojo"
    face_m, feat_m = face_masks(rgb.shape, r._faces[0])
    assert not hole[feat_m > 0.5].any() and not hole[face_m > 0.05].any(), "se tocó la cara"
    assert r.faces[0].check.passed or r.faces[0].applied == 0


def test_user_mask_on_face_is_smooth_not_invented():
    """Si se pinta sobre un ojo, el relleno es liso (sin estructura nueva)."""
    rgb = load("astronauta_original.png")
    face = FaceGuard.get().detect(rgb)[0]
    eye = face.landmarks[EYE_L].mean(0).astype(int)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    cv2.circle(mask, tuple(eye), int(face.interocular * 0.25), 255, -1)
    r = process(rgb, Settings(mode="restaurar", intensity=0.35, user_mask=mask))
    region = r.image[mask > 0].astype(np.float32)
    assert r.stats["filled_on_faces"] >= 1
    assert region.std(0).mean() < rgb[mask > 0].astype(np.float32).std(0).mean(), "el relleno del ojo no es liso"


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


def test_painting_restored_without_repainting():
    """Restaurar cuadro: se acerca al cuadro limpio, conserva los brillos
    pintados (no son lagunas) y no toca los rasgos de la cara."""
    from revo.painting import detect_losses

    clean, dirty = load("cuadro_original.png"), load("cuadro_deteriorado.png")
    res = process(dirty, Settings(mode="restaurar_cuadro", intensity=0.35))
    lab = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32)  # noqa: E731
    before = float(np.abs(lab(dirty) - lab(clean)).mean())
    after = float(np.abs(lab(res.image) - lab(clean)).mean())
    assert after < 0.8 * before, (before, after)
    # el brillo blanco del casco (pintado) no se propone como laguna
    losses, _ = detect_losses(dirty, 0.35)
    assert not losses[848:922, 689:745].any()
    # rasgos: iguales a la versión solo limpiada de barniz
    face = res._faces[0]
    _, feat = face_masks(dirty.shape, face)
    d = np.abs(res.image.astype(np.int16) - res._base.astype(np.int16)).max(-1)
    assert d[feat > 0.5].mean() < 2.0, d[feat > 0.5].mean()


def test_painting_cracks_ignore_fine_detail():
    """Pelo, encajes, cables o pinceladas finas no son grietas: en una imagen
    llena de detalle y sin craquelado casi nada se toma por grieta."""
    from revo.painting import detect_cracks

    for name in ("astronauta_original.png", "bodegon_original.png"):
        m = detect_cracks(load(name), 0.35)
        assert (m > 0).mean() < 0.02, (name, (m > 0).mean())
    # ni los reflejos pintados (cuchara, platillo) se toman por lagunas
    from revo.painting import detect_losses

    for name in ("cuadro_original.png", "bodegon_original.png"):
        assert detect_losses(load(name), 0.35)[1] == 0, name


def test_torn_portrait_rebuilt_from_its_own_paint():
    """Rotura que cruza la cara de un retrato pequeño: del trazo del pincel
    solo se repara lo dañado, la cara se reconstruye desde su lado sano y la
    rotura desaparece."""
    from revo.painting import refine_brush

    oil, torn = load("retrato_original.png"), load("retrato_roto.png")
    tear = cv2.imread(os.path.join(ROOT, "samples", "retrato_rotura.png"), 0) > 0
    brush = cv2.imread(os.path.join(ROOT, "samples", "retrato_pincel.png"), 0)
    m = refine_brush(torn, brush, 0.35) > 0
    assert m[tear].mean() > 0.95, m[tear].mean()
    assert m.sum() < 0.8 * (brush > 0).sum(), m.sum() / (brush > 0).sum()
    res = process(torn, Settings(mode="restaurar_cuadro", intensity=0.35, user_mask=brush))
    err = lambda x: np.abs(x.astype(np.int16) - oil.astype(np.int16)).mean(-1)  # noqa: E731
    assert err(res.image)[tear].mean() < 0.4 * err(torn)[tear].mean()
    assert len(res.faces) == 1 and res.faces[0].applied > 0
    assert res.stats["mirrored"] > 300, res.stats["mirrored"]


def test_flaking_painting_detected_but_clean_one_not():
    """Cuadro con cientos de desconchados (la preparación crema asoma): se
    encuentran casi todos; en el mismo cuadro sano, ninguno."""
    from revo.painting import detect_flakes

    clean = load("cuadro_original.png")
    assert not detect_flakes(clean, 0.35).any()
    rng = np.random.default_rng(5)
    img, gt = clean.copy(), np.zeros(clean.shape[:2], np.uint8)
    h, w = gt.shape
    for _ in range(600):
        cx, cy = int(rng.uniform(0.05, 0.95) * w), int(rng.uniform(0.05, 0.95) * h)
        r = int(rng.integers(2, 7))
        pts = np.array([[cx + rng.integers(-r, r + 1), cy + rng.integers(-r, r + 1)] for _ in range(5)], np.int32)
        cv2.fillPoly(gt, [cv2.convexHull(pts)], 1)
    img[gt > 0] = (232, 222, 200)
    m = detect_flakes(img, 0.35) > 0
    assert m[gt > 0].mean() > 0.6, m[gt > 0].mean()
    assert (m & ~cv2.dilate(gt, np.ones((7, 7), np.uint8)).astype(bool)).mean() < 0.04


def test_gemini_restoration_flow_without_network():
    """Restaurar cuadro con Gemini: se envía la foto con las instrucciones y
    el cuadro devuelto vuelve al tamaño original (sin red: respuesta simulada)."""
    import base64
    import json

    from revo import gemini
    from revo.pipeline import process_gemini, summary_lines

    img = load("cuadro_deteriorado.png")
    clean = load("cuadro_original.png")
    sent = {}

    def fake_post(model, key, body):
        sent.update(model=model, key=key, body=json.loads(body))
        small = cv2.resize(clean, (512, 512))
        ok, buf = cv2.imencode(".png", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        data = base64.b64encode(buf.tobytes()).decode()
        return {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": data}}]}}]}

    real = gemini._post
    gemini._post = fake_post
    try:
        res = process_gemini(img, Settings(mode="restaurar_cuadro"), key="AIzaPRUEBA")
    finally:
        gemini._post = real
    assert sent["key"] == "AIzaPRUEBA" and sent["model"] == gemini.MODELS[0]
    parts = sent["body"]["contents"][0]["parts"]
    assert "Restore" in parts[0]["text"] and parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    assert res.image.shape == img.shape
    assert any("Gemini" in line for line in summary_lines(res))
    try:
        gemini.restore(img, key="")
        raise AssertionError("sin clave debería fallar")
    except gemini.GeminiError as e:
        assert "clave" in str(e)


def test_cloud_chain_flux_then_qwen_without_network():
    """FLUX falla → Qwen responde; el resultado vuelve al tamaño original."""
    from revo import cloud
    from revo.pipeline import process_cloud

    img = np.full((300, 200, 3), 128, np.uint8)
    seen = []

    def fake_call(space, src):
        seen.append(space)
        if "FLUX" in space:
            raise RuntimeError("You have exceeded your GPU quota")
        return np.full((512, 341, 3), 90, np.uint8)

    try:
        process_cloud(img, Settings(mode="restaurar_cuadro"), "FLUX", call=fake_call)
        raise AssertionError("FLUX debía fallar")
    except cloud.CloudError as e:
        assert "agotado" in str(e)
    res = process_cloud(img, Settings(mode="restaurar_cuadro"), "Qwen", call=fake_call)
    assert res.image.shape == img.shape and res.stats["cloud"] == "Qwen"
    assert seen[0].startswith("black-forest-labs") and seen[1].startswith("Qwen/")


def test_cloud_fills_space_parameters_by_name():
    from revo import cloud

    class FakeClient:
        def view_api(self, **_):
            return {"named_endpoints": {"/infer": {"parameters": [
                {"parameter_name": "image", "component": "Image"},
                {"parameter_name": "prompt", "component": "Textbox"},
                {"parameter_name": "seed", "component": "Slider"},
                {"parameter_name": "rewrite_prompt", "component": "Checkbox"},
                {"parameter_name": "num_inference_steps", "component": "Slider"},
            ]}}}

    name, kw = cloud._arguments(FakeClient(), __file__)
    assert name == "/infer" and kw["prompt"] == cloud.PROMPT
    assert kw["seed"] == 0 and kw["rewrite_prompt"] is False and "num_inference_steps" not in kw
    assert cloud._find_image(([{"image": {"path": __file__}}], 0)) == __file__


def test_repaint_only_inside_the_zone_and_blends():
    """La IA de pintar solo cambia la zona perdida (y su borde fundido)."""
    from revo import repaint

    img = np.full((200, 300, 3), 100, np.uint8)
    zones = np.zeros((200, 300), np.uint8)
    zones[80:120, 100:160] = 255
    calls = []

    def fake_paint(crop, hole, seed):
        calls.append(crop.shape)
        return np.full_like(crop, 200)

    out, n = repaint.repaint(img, zones, paint=fake_paint)
    assert n == 1 and len(calls) == 1
    assert (out[90:110, 110:150] == 200).all()
    assert (out[:40] == 100).all() and (out[:, :40] == 100).all()


def test_tap_zone_follows_the_touched_object_and_spares_faces():
    """Un toque dentro de una mancha clara la coge entera, no el fondo; y
    nunca una cara."""
    from revo import painting

    rgb = np.full((300, 300, 3), 40, np.uint8)
    cv2.circle(rgb, (150, 150), 18, (230, 225, 215), -1)
    taps = np.zeros((300, 300), np.uint8)
    cv2.circle(taps, (150, 150), 4, 255, -1)
    z = painting.tap_zones(rgb, taps) > 0
    assert z[150, 135] and z[150, 165] and not z[150, 172] and not z[20, 20]
    face = np.zeros((300, 300), np.float32)
    face[100:200, 100:200] = 1
    assert not (painting.tap_zones(rgb, taps, face) > 0).any()


def test_lama_packs_small_crops_four_per_pass():
    from revo import inpaint
    calls = []

    def fake(crop, hole):
        calls.append(crop.shape[:2])
        out = crop.copy()
        out[hole > 0] = (0, 200, 0)
        return out

    img = np.full((600, 600, 3), 120, np.uint8)
    jobs = []
    for k in range(5):
        x0, y0 = 20 + 100 * k, 30
        hole = np.zeros((150, 90), np.uint8)
        hole[60:90, 30:60] = 1
        jobs.append(((x0, y0, x0 + 90, y0 + 150), hole > 0, hole))
    jobs.append(((0, 200, 500, 600), None, np.ones((400, 500), np.uint8)))
    old = inpaint._lama_fill
    inpaint._lama_fill = fake
    try:
        recs = inpaint._lama_many(img, jobs)
    finally:
        inpaint._lama_fill = old
    assert len(calls) == 3, calls  # el grande solo, y 5 pequeños en 2 pasadas
    for (_, _, hole), rec in zip(jobs[:5], recs[:5]):
        assert rec.shape[:2] == hole.shape
        assert rec[75, 45, 1] > 180 and abs(int(rec[10, 10, 1]) - 120) < 10


def test_sweep_leaves_no_specks_but_spares_protected_and_edges():
    from revo import painting
    rng = np.random.default_rng(1)
    img = np.zeros((400, 400, 3), np.uint8)
    img[:] = (170, 30, 45)  # vestido rojo
    img += rng.integers(0, 6, img.shape, dtype=np.uint8)
    cv2.circle(img, (300, 300), 70, (235, 225, 210), -1)  # gato blanco
    dots = [(40, 40), (80, 120), (150, 60), (60, 200)]
    for x, y in dots:
        cv2.circle(img, (x, y), 2, (240, 235, 225), -1)
    for x, y in [(290, 290), (310, 320)]:
        cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
    protect = np.zeros((400, 400), bool)
    protect[180:220, 40:80] = True  # una cara: su mota se queda
    out, touched = painting.sweep_specks(img, protect)
    for x, y in dots[:3]:
        assert out[y, x, 1] < 90, (x, y, out[y, x])  # la mota se pintó de rojo
    assert (out[200, 60] == img[200, 60]).all()  # la cara, intacta
    assert out[290, 290].astype(int).sum() < 255 * 3 - 20  # moteado igualado
    assert out[300, 368, 1] > 180  # el borde del gato sigue siendo blanco
    assert not touched[protect].any()


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
