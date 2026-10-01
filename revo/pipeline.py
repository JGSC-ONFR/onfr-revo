"""Pipeline principal de ONFR REVO."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import enhance, restore
from .analysis import Analysis, analyze, estimate_noise
from .faces import Face, FaceGuard, face_masks
from .fidelity import FaceCheck, check_face, identity_limit

MODES = {
    "mejorar": "Mejorar",
    "restaurar": "Restaurar",
    "mejorar_restaurar": "Mejorar + Restaurar",
    "restaurar_cuadro": "Restaurar cuadro",
}
PHASE_2 = {"restaurar_cuadro"}

# intensidades que se prueban sobre una cara hasta pasar la verificación;
# 0 = cara original, solo escalada
FACE_STEPS = (0.7, 0.5, 0.35, 0.2, 0.1, 0.0)


@dataclass
class Settings:
    mode: str = "mejorar_restaurar"
    intensity: float = 0.35  # 0 = conservadora, 1 = intensa
    scale: int = 2  # solo en modos con "Mejorar"
    colorize: bool = False  # fase 2
    grain: float | None = None  # None = automático según el modo

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
    intervention_map: np.ndarray
    analysis: Analysis
    settings: Settings
    faces: list[FaceReport]
    stats: dict = field(default_factory=dict)


def intensity_label(i: float) -> str:
    return "Conservadora" if i < 0.4 else ("Moderada" if i < 0.7 else "Intensa")


def process(rgb: np.ndarray, settings: Settings, progress=None) -> Result:
    t0 = time.time()
    say = progress or (lambda frac, msg: None)
    if settings.mode in PHASE_2:
        raise NotImplementedError("«Restaurar cuadro» llegará en la fase 2.")
    if settings.colorize:
        raise NotImplementedError("La colorización llegará en la fase 2.")

    i = float(np.clip(settings.intensity, 0, 1))
    scale = settings.scale if settings.enhances else 1
    rgb = np.ascontiguousarray(rgb[..., :3])
    h, w = rgb.shape[:2]

    say(0.05, "Analizando imagen")
    ana = analyze(rgb)

    say(0.15, "Detectando rostros")
    guard = FaceGuard.get()
    faces = guard.detect(rgb)
    face_u = np.zeros((h, w), np.float32)
    feat_u = np.zeros((h, w), np.float32)
    for f in faces:
        fm, ft = face_masks(rgb.shape, f)
        face_u, feat_u = np.maximum(face_u, fm), np.maximum(feat_u, ft)

    # 1) defectos (solo restauración)
    defects = np.zeros((h, w), np.uint8)
    defect_report = {"skipped_on_faces": 0}
    work = rgb
    if settings.restores:
        say(0.25, "Localizando polvo y arañazos")
        defects = restore.detect_defects(rgb, ana.noise, face_u, feat_u, i, defect_report)
        work = restore.repair_defects(rgb, defects)

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

    say(0.55, "Corrigiendo luz y contraste")
    tone_i = i if settings.restores else 0.5 * i
    lut = restore.tone_lut(work, tone_i)
    gen = restore.apply_tone(gen, lut)
    gen = restore.local_contrast(gen, tone_i)

    say(0.65, "Aumentando resolución" if scale > 1 else "Afinando")
    gen = enhance.upscale(gen, scale)
    if settings.enhances:
        gen = enhance.sharpen(gen, 0.5 + 1.0 * i, scale, estimate_noise(cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY)))

    # versión mínima: solo defectos seguros + tono global + escalado
    minimal_small = restore.apply_tone(work, lut)
    base = enhance.upscale(minimal_small, scale)

    # 3) protección facial con verificación
    out = gen.copy()
    reports: list[FaceReport] = []
    ignore = cv2.dilate(defects, np.ones((5, 5), np.uint8))
    for n, f in enumerate(faces):
        say(0.75 + 0.15 * n / max(1, len(faces)), f"Verificando rostro {n + 1}")
        fm_small, _ = face_masks(rgb.shape, f)
        fm_big, _ = face_masks(out.shape, f, scale)
        m = fm_big[..., None]
        # referencia = la cara con la intervención mínima (tono global y motas
        # seguras); cualquier cambio de rasgos o identidad se mide contra ella
        ref = guard.describe(minimal_small, Face(f.rect, f.landmarks.copy()))
        id_limit = identity_limit(guard, ref, minimal_small, ana.noise)
        for k, a in enumerate(FACE_STEPS):
            face_px = base.astype(np.float32) + a * (gen.astype(np.float32) - base.astype(np.float32))
            cand = np.clip(out * (1 - m) + face_px * m, 0, 255).astype(np.uint8)
            small = cand if scale == 1 else cv2.resize(cand, (w, h), interpolation=cv2.INTER_AREA)
            chk = check_face(guard, ref, small, minimal_small, fm_small, ignore, id_limit)
            if chk.passed or a == 0:
                out = cand
                reports.append(FaceReport(n + 1, f.rect, a, chk, k + 1))
                break

    say(0.95, "Generando resumen")
    before = enhance.upscale(rgb, scale)
    imap, changed = intervention_map(base, out, defects, faces, reports, scale)
    stats = {
        "seconds": time.time() - t0,
        "defect_pixels": int((defects > 0).sum()),
        "defect_regions": int(cv2.connectedComponents((defects > 0).astype(np.uint8))[0] - 1),
        "changed_fraction": changed,
        "grain_kept": grain,
        "skipped_on_faces": defect_report["skipped_on_faces"],
    }
    return Result(out, before, imap, ana, settings, reports, stats)


def intervention_map(before, after, defects, faces: list[Face], reports, scale):
    """Mapa visual de qué se ha tocado. `before` es la versión mínima (solo
    ajuste global de tono), así el calor muestra los retoques locales:
    ruido, nitidez y contraste local. Magenta = defectos rellenados,
    recuadros = caras protegidas."""
    Lb = cv2.cvtColor(before, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    La = cv2.cvtColor(after, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    d = cv2.GaussianBlur(np.abs(La - Lb), (0, 0), 1.5 * scale)
    changed = float((d > 6).mean())
    heat = cv2.applyColorMap(np.clip(d * 255 / 30, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(cv2.cvtColor(before, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(gray, 0.45, heat, 0.55, 0)
    if defects.any():
        dm = cv2.resize(defects, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        dm = cv2.dilate(dm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        img[dm] = (255, 0, 255)
    t = max(1, img.shape[1] // 400)
    for f, r in zip(faces, reports):
        x0, y0, x1, y1 = [int(v * scale) for v in f.rect]
        col = (60, 220, 90) if r.applied > 0 else (80, 170, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, t)
        cv2.putText(img, f"Cara {r.index}", (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, col, t)
    return img, changed


def summary_lines(res: Result) -> list[str]:
    s, a = res.settings, res.analysis
    oh, ow = a.height, a.width
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
        "Colorización: Desactivada  ",
        "Protección facial: Activada  ",
        f"Resolución: {ow} × {oh} → {nw} × {nh}",
    ]
    lines.append("")
    lines.append("**Zonas tocadas**")
    if s.restores:
        n = res.stats["defect_regions"]
        pct = 100 * res.stats["defect_pixels"] / (ow * oh)
        lines.append(f"- Defectos reparados: {n} (polvo, motas, arañazos finos; {pct:.2f}% de la imagen)")
        if res.stats["skipped_on_faces"]:
            lines.append(
                f"- Marcas sobre rostros que se han dejado intactas: {res.stats['skipped_on_faces']} "
                "(sobre rasgos nunca se reconstruye; las manchas oscuras pueden ser lunares)"
            )
        lines.append(f"- Grano original conservado: {res.stats['grain_kept']:.0%}")
    lines.append("- Luz y contraste: ajuste global moderado (igual en toda la imagen)")
    lines.append(f"- Retoques locales apreciables (ruido, nitidez, contraste local): {100 * res.stats['changed_fraction']:.1f}% de la imagen")
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
    lines.append(f"_Tiempo: {res.stats['seconds']:.1f} s_")
    return lines
