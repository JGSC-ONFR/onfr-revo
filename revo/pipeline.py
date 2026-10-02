"""Pipeline principal de ONFR REVO."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import colorize as colorizer
from . import enhance, inpaint, painting, restore, symmetry
from .analysis import Analysis, analyze, estimate_noise
from .faces import Face, FaceGuard, face_masks
from .fidelity import FaceCheck, check_face, identity_limit

MODES = {
    "mejorar": "Mejorar",
    "restaurar": "Restaurar",
    "mejorar_restaurar": "Mejorar + Restaurar",
    "restaurar_cuadro": "Restaurar cuadro",
}
PHASE_2: set[str] = set()

# intensidades que se prueban sobre una cara hasta pasar la verificación;
# 0 = cara original, solo escalada
FACE_STEPS = (0.7, 0.5, 0.35, 0.2, 0.1, 0.0)
MAX_WORK = 2000  # px del lado largo con que se trabaja
MAX_OUT = 4000  # px del lado largo del resultado


@dataclass
class Settings:
    mode: str = "mejorar_restaurar"
    intensity: float = 0.35  # 0 = conservadora, 1 = intensa
    scale: int = 2  # solo en modos con "Mejorar"
    colorize: bool = False  # colorización opcional (estimación)
    color_amount: float = 0.8  # intensidad del color estimado
    grain: float | None = None  # None = automático según el modo
    user_mask: np.ndarray | None = None  # daños marcados con el pincel (>0)
    auto_damage: bool = True  # detectar roturas y marcas grandes (la interfaz las propone en el pincel)

    @property
    def restores(self) -> bool:
        return self.mode in ("restaurar", "mejorar_restaurar")

    @property
    def enhances(self) -> bool:
        return self.mode in ("mejorar", "mejorar_restaurar")


@dataclass
class FaceReport:
    index: int
    rect: tuple
    applied: float  # fracción de la mejora aplicada a la cara (0-1)
    check: FaceCheck
    attempts: int

    @property
    def status(self) -> str:
        if self.applied == 0:
            return "conservada como el original (no se pudo mejorar con seguridad)"
        if self.applied < FACE_STEPS[0]:
            return f"mejora reducida al {self.applied:.0%} para no alterarla"
        return "mejorada de forma conservadora"


@dataclass
class Result:
    image: np.ndarray
    before: np.ndarray  # original escalado al mismo tamaño (para comparar)
    analysis: Analysis
    settings: Settings
    faces: list[FaceReport]
    stats: dict = field(default_factory=dict)
    # datos para el resumen y el mapa, que se generan solo si se piden
    _base: np.ndarray | None = None
    _uncolored: np.ndarray | None = None
    _masks: dict = field(default_factory=dict)
    _faces: list = field(default_factory=list)


def intensity_label(i: float) -> str:
    return "Conservadora" if i < 0.4 else ("Moderada" if i < 0.7 else "Intensa")


def _crop_box(face: Face, shape, pad=0.6):
    x0, y0, x1, y1 = face.rect
    p = int(pad * (x1 - x0))
    h, w = shape[:2]
    return max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)


def _shift(face: Face, dx, dy) -> Face:
    x0, y0, x1, y1 = face.rect
    return Face((x0 - dx, y0 - dy, x1 - dx, y1 - dy), face.landmarks - [dx, dy], face.embedding)


def _detect_small(guard: FaceGuard, rgb: np.ndarray) -> list[Face]:
    """Caras de un cuadro pequeño: se buscan sobre una copia ampliada (el
    detector no ve caras de menos de ~80 px) y se devuelven a su tamaño."""
    h, w = rgb.shape[:2]
    f = 1.0 if max(h, w) >= 700 else 700 / max(h, w)
    if f == 1.0:
        return guard.detect(rgb, embed=False)
    big = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    out = []
    for fc in guard.detect(big, embed=False):
        x0, y0, x1, y1 = fc.rect
        r = tuple(int(round(v / f)) for v in (x0, y0, x1, y1))
        out.append(Face(r, fc.landmarks / f))
    return out


def process(rgb: np.ndarray, settings: Settings, progress=None) -> Result:
    t0 = time.time()
    say = progress or (lambda frac, msg: None)
    if settings.mode in PHASE_2:
        raise NotImplementedError("«Restaurar cuadro» llegará en la fase 2.")
    if settings.colorize and not colorizer.available():
        raise RuntimeError("Falta el modelo de color. Ejecuta: python setup_models.py")

    if settings.mode == "restaurar_cuadro":
        return _process_painting(rgb, settings, say, t0)

    i = float(np.clip(settings.intensity, 0, 1))
    scale = settings.scale if settings.enhances else 1
    rgb = np.ascontiguousarray(rgb[..., :3])
    orig_h, orig_w = rgb.shape[:2]
    # tamaño de trabajo acotado: una foto de móvil (4000 px) tardaría más de
    # un minuto y no aporta detalle real; el resultado tampoco pasa de MAX_OUT
    f = min(1.0, MAX_WORK / max(orig_h, orig_w))
    if f < 1:
        rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    scale = max(1, min(scale, MAX_OUT // max(h, w)))

    say(0.05, "Analizando imagen")
    ana = analyze(rgb)

    say(0.12, "Detectando rostros")
    guard = FaceGuard.get()
    faces = guard.detect(rgb, embed=False)  # la identidad se mide luego sobre el recorte
    face_u = np.zeros((h, w), np.float32)
    feat_u = np.zeros((h, w), np.float32)
    for f in faces:
        fm, ft = face_masks(rgb.shape, f)
        face_u, feat_u = np.maximum(face_u, fm), np.maximum(feat_u, ft)

    # 1) daños: motas y arañazos finos, roturas y marcas, y lo pintado a mano
    empty = np.zeros((h, w), np.uint8)
    defects, damage, user = empty, empty, empty
    defect_report = {"skipped_on_faces": 0}
    fill_info = {"small": 0, "large": 0, "on_faces": 0, "engine": "FSR"}
    stains = 0.0
    work = rgb
    if settings.user_mask is not None and settings.user_mask.any():
        um = settings.user_mask
        if um.shape[:2] != (h, w):
            um = cv2.resize(um.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        user = (um > 0).astype(np.uint8) * 255
    if settings.restores:
        say(0.22, "Localizando polvo, arañazos y roturas")
        defects = restore.detect_defects(rgb, ana.noise, face_u, feat_u, i, defect_report)
        if settings.auto_damage:
            damage, _ = restore.detect_damage(rgb, ana.monochrome, face_u, feat_u, i)
    holes = ((defects > 0) | (damage > 0) | (user > 0)).astype(np.uint8)
    if holes.any():
        say(0.3, "Reparando daños")
        work, fill_info = inpaint.fill(rgb, holes, face_u)
    if settings.restores and ana.monochrome:
        work, stains = restore.retone_stains(work, i, face_u)

    # 2) cadena general
    say(0.4, "Reduciendo ruido")
    if settings.grain is not None:
        grain = settings.grain
    elif settings.mode == "restaurar":
        grain = 0.15 + 0.4 * (1 - i)
    elif settings.mode == "mejorar_restaurar":
        grain = 0.1 + 0.25 * (1 - i)
    else:
        grain = 0.1 * (1 - i)
    denoise_needed = settings.restores or ana.noise > 2.5 or ana.jpeg_blockiness > 1.25
    gen = restore.denoise(work, ana.noise, i, ana.monochrome) if denoise_needed else work.copy()
    gen = restore.keep_grain(work, gen, grain)

    say(0.5, "Corrigiendo luz y contraste")
    tone_i = i if settings.restores else 0.5 * i
    lut = restore.tone_lut(work, tone_i)
    gen = restore.apply_tone(gen, lut)
    gen = restore.local_contrast(gen, tone_i)

    say(0.6, "Aumentando resolución" if scale > 1 else "Afinando")
    gen = enhance.upscale(gen, scale)
    if settings.enhances:
        gen = enhance.sharpen(gen, 0.5 + 1.0 * i, scale, estimate_noise(cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)))

    # versión mínima: solo daños reparados + tono global + escalado
    minimal_small = restore.apply_tone(work, lut)
    base = enhance.upscale(minimal_small, scale)

    # 3) protección facial con verificación (sobre un recorte de cada cara)
    out = gen.copy()
    reports: list[FaceReport] = []
    ignore = cv2.dilate(holes * 255, np.ones((5, 5), np.uint8))
    for n, f in enumerate(faces):
        say(0.7 + 0.2 * n / max(1, len(faces)), f"Verificando rostro {n + 1}")
        X0, Y0, X1, Y1 = _crop_box(f, rgb.shape)
        sx0, sy0, sx1, sy1 = X0 * scale, Y0 * scale, X1 * scale, Y1 * scale
        ref_img = np.ascontiguousarray(minimal_small[Y0:Y1, X0:X1])
        fl = _shift(f, X0, Y0)
        fm_small, _ = face_masks(ref_img.shape, fl)
        fm_big, _ = face_masks((sy1 - sy0, sx1 - sx0), fl, scale)
        m = fm_big[..., None]
        # referencia = la cara con la intervención mínima; cualquier cambio
        # de rasgos o identidad se mide contra ella
        ref = guard.describe(ref_img, Face(fl.rect, fl.landmarks.copy()))
        id_limit = identity_limit(guard, ref, ref_img, ana.noise)
        b_crop = base[sy0:sy1, sx0:sx1].astype(np.float32)
        g_crop = gen[sy0:sy1, sx0:sx1].astype(np.float32)
        o_crop = out[sy0:sy1, sx0:sx1].astype(np.float32)
        for k, a in enumerate(FACE_STEPS):
            face_px = b_crop + a * (g_crop - b_crop)
            cand = np.clip(o_crop * (1 - m) + face_px * m, 0, 255).astype(np.uint8)
            small = cand if scale == 1 else cv2.resize(cand, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_AREA)
            chk = check_face(guard, ref, small, ref_img, fm_small, ignore[Y0:Y1, X0:X1], id_limit, lazy=a > 0)
            if chk.passed or a == 0:
                out[sy0:sy1, sx0:sx1] = cand
                reports.append(FaceReport(n + 1, f.rect, a, chk, k + 1))
                break

    uncolored = None
    if settings.colorize:
        say(0.93, "Estimando colores")
        uncolored = out
        out = colorizer.colorize(out, settings.color_amount)

    stats = {
        "seconds": time.time() - t0,
        "original_size": (orig_w, orig_h),
        "defect_regions": int(cv2.connectedComponents((defects > 0).astype(np.uint8))[0] - 1),
        "damage_regions": int(cv2.connectedComponents((damage > 0).astype(np.uint8))[0] - 1),
        "user_regions": int(cv2.connectedComponents((user > 0).astype(np.uint8))[0] - 1),
        "filled_pixels": int(holes.sum()),
        "defect_pixels": int((defects > 0).sum()),
        "filled_on_faces": fill_info["on_faces"],
        "engine": "IA de relleno (LaMa)" if fill_info.get("engine") == "LaMa" else "la textura de alrededor",
        "stains": stains,
        "grain_kept": grain,
        "skipped_on_faces": defect_report["skipped_on_faces"],
    }
    return Result(
        out, enhance.upscale(rgb, scale), ana, settings, reports, stats,
        _base=base, _uncolored=uncolored,
        _masks={"defects": defects, "damage": damage, "user": user}, _faces=faces,
    )


def _process_painting(rgb: np.ndarray, settings: Settings, say, t0: float) -> Result:
    """Restaurar cuadro (ver painting.py): barniz y suciedad, craquelado,
    lagunas y zonas descoloridas. Sin reducir ruido, enfocar ni ampliar: la
    pincelada y la textura se quedan como están."""
    i = float(np.clip(settings.intensity, 0, 1))
    rgb = np.ascontiguousarray(rgb[..., :3])
    orig_h, orig_w = rgb.shape[:2]
    f = min(1.0, MAX_WORK / max(orig_h, orig_w))
    if f < 1:
        rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]

    say(0.05, "Analizando el cuadro")
    ana = analyze(rgb)
    say(0.1, "Localizando los daños")
    empty = np.zeros((h, w), np.uint8)
    user = empty
    if settings.user_mask is not None and settings.user_mask.any():
        um = settings.user_mask
        if um.shape[:2] != (h, w):
            um = cv2.resize(um.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        # del trazo del pincel, solo lo que está dañado de verdad
        user = painting.refine_brush(rgb, (um > 0).astype(np.uint8) * 255, i)
    rough = painting.detect_losses(rgb, i)[0] if settings.auto_damage else empty
    rough_holes = cv2.dilate(((rough > 0) | (user > 0)).astype(np.uint8), np.ones((5, 5), np.uint8))

    say(0.15, "Quitando barniz amarillento y suciedad")
    clean, varnish = painting.clean_varnish(rgb, i)
    # versión con los huecos cerrados a lo liso: con ella se encuentran las
    # caras aunque la rotura las cruce, y es la referencia de sus rasgos
    quick = cv2.inpaint(clean, rough_holes, 3, cv2.INPAINT_TELEA) if rough_holes.any() else clean

    say(0.2, "Buscando rostros")
    guard = FaceGuard.get()
    faces = _detect_small(guard, quick)
    face_u = np.zeros((h, w), np.float32)
    feat_u = np.zeros((h, w), np.float32)
    for fc in faces:
        fm, ft = face_masks(rgb.shape, fc)
        face_u, feat_u = np.maximum(face_u, fm), np.maximum(feat_u, ft)

    say(0.3, "Cerrando grietas")
    work = clean
    cracks = painting.detect_cracks(work, i, feat_u)
    if cracks.any():
        work = cv2.inpaint(work, cracks, 3, cv2.INPAINT_TELEA)

    say(0.45, "Reintegrando lagunas")
    losses = painting.detect_losses(rgb, i, feat_u)[0] if settings.auto_damage else empty
    holes = ((losses > 0) | (user > 0)).astype(np.uint8)
    fill_info = {"on_faces": 0, "engine": "FSR"}
    plain = work
    if holes.any():
        plain, fill_info = inpaint.fill(work, holes, face_u)
    work = plain

    # caras: lo que falta de un lado se toma del otro lado, sano, del propio
    # cuadro (reflejado, alineado con sus puntos y con la luz del sitio)
    sym = np.zeros((h, w), bool)
    if holes.any() and faces:
        say(0.55, "Reconstruyendo la cara a partir de su lado sano")
        hole_d = cv2.dilate(holes, np.ones((3, 3), np.uint8))
        for fc in faces:
            fm, _ = face_masks(rgb.shape, fc)
            work, done = symmetry.mirror_fill(work, hole_d, fm, fc.landmarks, fc.interocular)
            sym |= done

    say(0.7, "Recuperando el color perdido")
    work, faded = painting.revive_faded(work, i, feat_u)
    plain_f = painting.revive_faded(plain, i, feat_u)[0] if sym.any() else work

    # rostros: si algo de lo anterior hubiera movido un rasgo, se deshace:
    # primero la reconstrucción por simetría, y si aun así no cuadra, la cara
    # vuelve a la versión solo limpiada de barniz
    reports: list[FaceReport] = []
    ignore = cv2.dilate(((holes > 0) | (cracks > 0)).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    for n, fc in enumerate(faces):
        say(0.8, f"Verificando rostro {n + 1}")
        X0, Y0, X1, Y1 = _crop_box(fc, rgb.shape)
        ref_img = np.ascontiguousarray(quick[Y0:Y1, X0:X1])
        fl = _shift(fc, X0, Y0)
        fm_small, _ = face_masks(ref_img.shape, fl)
        # aquí solo cuenta la forma (puntos y estructura): sin vector de identidad
        ref = Face(fl.rect, guard.landmarks(ref_img, fl.rect))
        id_limit = 0.0
        applied = 1.0
        for cand_full in ((work, plain_f) if sym[Y0:Y1, X0:X1].any() else (work,)):
            cand = np.ascontiguousarray(cand_full[Y0:Y1, X0:X1])
            chk = check_face(guard, ref, cand, ref_img, fm_small, ignore[Y0:Y1, X0:X1], id_limit, skip_hidden=True)
            # las grietas y los huecos de la referencia confunden al vector de
            # identidad, así que se exige lo que mide la forma
            if chk.landmarks <= 0.03 and chk.structure <= 3.0:
                break
        else:
            # ni así: la cara se queda limpia de barniz y con los huecos
            # cerrados a lo liso, sin volver a enseñar el daño
            cand = np.ascontiguousarray(quick[Y0:Y1, X0:X1])
            applied = 0.0
        if cand_full is not work or applied == 0.0:
            m = fm_small[..., None]
            cur = work[Y0:Y1, X0:X1].astype(np.float32)
            work[Y0:Y1, X0:X1] = np.clip(cur * (1 - m) + cand * m, 0, 255).astype(np.uint8)
            sym[Y0:Y1, X0:X1] &= fm_small < 0.5
        reports.append(FaceReport(n + 1, fc.rect, applied, chk, 1))

    uncolored = None
    if settings.colorize:
        say(0.93, "Estimando colores")
        uncolored = work
        work = colorizer.colorize(work, settings.color_amount)

    stats = {
        "seconds": time.time() - t0,
        "original_size": (orig_w, orig_h),
        "cracks": int((cracks > 0).mean() * 10000) / 100,
        "loss_regions": int(cv2.connectedComponents((losses > 0).astype(np.uint8))[0] - 1),
        "user_regions": int(cv2.connectedComponents((user > 0).astype(np.uint8))[0] - 1),
        "faded": faded,
        "varnish": varnish["removed"],
        "filled_on_faces": fill_info.get("on_faces", 0),
        "mirrored": int(sym.sum()),
        "engine": "IA de relleno (LaMa)" if fill_info.get("engine") == "LaMa" else "la textura de alrededor",
    }
    return Result(
        work, rgb.copy(), ana, settings, reports, stats,
        _base=clean, _uncolored=uncolored,
        _masks={"defects": cracks, "damage": losses, "user": user, "mirrored": sym}, _faces=faces,
    )


def _painting_lines(res: Result, changed: float) -> list[str]:
    s, st = res.settings, res.stats
    ow, oh = st["original_size"]
    nh, nw = res.image.shape[:2]
    da, db = st["varnish"]
    lines = [
        "**Restauración del cuadro completada**  ",
        f"Intervención: {intensity_label(s.intensity)}  ",
        "Pincelada, textura y composición: sin tocar (no se suaviza, ni se enfoca, ni se amplía)  ",
        f"Resolución: {ow} × {oh} → {nw} × {nh}",
        "",
        "**Zonas tocadas**",
        f"- Barniz amarillento y suciedad: corrección global (la misma en todo el cuadro), {abs(db):.0f} puntos menos de amarillo",
        f"- Grietas del craquelado cerradas: {st['cracks']:.2f}% del cuadro",
    ]
    if st["loss_regions"] or st["user_regions"]:
        lines.append(
            f"- Lagunas reintegradas con {st['engine']}: {st['loss_regions'] + st['user_regions']} "
            "(lo que había debajo no se puede recuperar; se continúa lo de alrededor)"
        )
    if st.get("mirrored"):
        lines.append(
            "- Cara: lo que faltaba se ha reconstruido a partir de su lado sano, reflejado y con la luz del sitio "
            "(en amarillo en el mapa). No se ha generado nada nuevo; lo que estaba en el eje o dañado a los dos "
            "lados se ha cerrado liso"
        )
    if st["faded"] > 0:
        lines.append(
            f"- Zonas descoloridas: {100 * st['faded']:.1f}% del cuadro. Se aviva el color que aún conservaban, "
            "hasta acercarlo al de su entorno; donde se perdió del todo no se inventa (la luz y la pincelada no cambian)"
        )
    if s.colorize:
        lines.append("- Color: estimado por IA. Solo se añade color; la forma y la luz son las del cuadro")
    for r in res.faces:
        lines.append(f"- Rostro {r.index}: " + ("verificado, los rasgos sanos no se han movido" if r.applied else "sin reconstruir (no cuadraba con sus rasgos): solo limpio y con los huecos cerrados a lo liso"))
    lines.append(f"- Retoques locales apreciables: {100 * changed:.1f}% del cuadro")
    lines.append("")
    lines.append(f"_Tiempo: {st['seconds']:.1f} s_")
    return lines


def intervention_map(res: Result) -> tuple[np.ndarray, float]:
    """Mapa visual de qué se ha tocado. Calor = retoques locales (ruido,
    nitidez, contraste local) frente a la versión mínima; magenta = daños
    rellenados automáticamente; cian = zonas marcadas con el pincel;
    recuadros = caras protegidas."""
    before = res._base
    after = res._uncolored if res._uncolored is not None else res.image
    scale = after.shape[1] / res.analysis.width
    Lb = cv2.cvtColor(before, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    La = cv2.cvtColor(after, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    d = cv2.GaussianBlur(np.abs(La - Lb), (0, 0), 1.5 * scale)
    changed = float((d > 6).mean())
    heat = cv2.applyColorMap(np.clip(d * 255 / 30, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(cv2.cvtColor(res.before, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(gray, 0.45, heat, 0.55, 0)
    size = (img.shape[1], img.shape[0])
    for key, col in (("defects", (255, 0, 255)), ("damage", (255, 0, 255)), ("user", (0, 220, 255)),
                     ("mirrored", (255, 210, 0))):
        mk = res._masks.get(key)
        if mk is not None and mk.any():
            dm = cv2.resize(mk, size, interpolation=cv2.INTER_NEAREST) > 0
            img[cv2.dilate(dm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = col
    t = max(1, img.shape[1] // 400)
    for f, r in zip(res._faces, res.faces):
        x0, y0, x1, y1 = [int(v * scale) for v in f.rect]
        col = (60, 220, 90) if r.applied > 0 else (80, 170, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, t)
        cv2.putText(img, f"Cara {r.index}", (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, col, t)
    return img, changed


def summary_lines(res: Result, changed: float | None = None) -> list[str]:
    s, a, st = res.settings, res.analysis, res.stats
    if changed is None:
        changed = intervention_map(res)[1]
    if s.mode == "restaurar_cuadro":
        return _painting_lines(res, changed)
    ow, oh = st.get("original_size", (a.width, a.height))
    nh, nw = res.image.shape[:2]
    title = {
        "mejorar": "Mejora completada",
        "restaurar": "Restauración completada",
        "mejorar_restaurar": "Mejora y restauración completadas",
    }[s.mode]
    lines = [
        f"**{title}**  ",
        f"Modo: {MODES[s.mode]}  ",
        f"Intervención: {intensity_label(s.intensity)}  ",
        ("Colorización: Activada (colores estimados, no recuperados)  " if s.colorize else "Colorización: Desactivada  "),
        "Protección facial: Activada  ",
        f"Resolución: {ow} × {oh} → {nw} × {nh}",
        "",
        "**Zonas tocadas**",
    ]
    if s.restores:
        lines.append(f"- Polvo, motas y arañazos finos reparados: {st['defect_regions']}")
        if st["damage_regions"]:
            lines.append(f"- Roturas y marcas grandes rellenadas con {st['engine']}: {st['damage_regions']}")
        if st["stains"] > 0.001:
            lines.append(f"- Manchas de color devueltas al tono de la foto: {100 * st['stains']:.1f}% de la imagen (solo color)")
        if st["skipped_on_faces"]:
            lines.append(
                f"- Marcas sobre rostros que se han dejado intactas: {st['skipped_on_faces']} "
                "(sobre rasgos nunca se reconstruye; las manchas oscuras pueden ser lunares)"
            )
        lines.append(f"- Grano original conservado: {st['grain_kept']:.0%}")
    if st["user_regions"]:
        lines.append(
            f"- Zonas marcadas con el pincel (sugeridas o pintadas): {st['user_regions']}. "
            f"Rellenadas con {st['engine']}; lo que había debajo no se puede recuperar"
        )
    if st["filled_on_faces"]:
        lines.append("- Sobre la cara el relleno es liso: no se reconstruyen ojos, nariz ni boca")
    lines.append("- Luz y contraste: ajuste global moderado (igual en toda la imagen)")
    lines.append(f"- Retoques locales apreciables (ruido, nitidez, contraste local): {100 * changed:.1f}% de la imagen")
    if s.colorize:
        lines.append("- Color: estimado por IA. Solo se añade color; la forma, los rasgos y la luz son los de la foto")
    if not res.faces:
        lines.append("- Rostros: no se han detectado")
    for r in res.faces:
        c = r.check
        ident = f"{c.identity:.3f}" if c.identity is not None else "n/d"
        lines.append(
            f"- Cara {r.index}: {r.status}. Cambio de identidad {ident} (límite {c.identity_limit:.2f}) · "
            f"rasgos {100 * c.landmarks:.1f}% (límite 3%) · estructura {c.structure:.1f} (límite 3)"
        )
    lines.append("")
    lines.append(f"_Tiempo: {st['seconds']:.1f} s_")
    return lines
