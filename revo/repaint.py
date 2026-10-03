"""Repintado de lo que se ha perdido del todo (IA generativa local).

Donde un cuadro ha perdido la pintura por completo (la cara de un gato de la
que solo quedan dos manchitas de los ojos), restaurar no basta: no hay nada
debajo que recuperar. Ahí, y solo donde se toca con el pincel, REVO usa
Stable Diffusion Inpainting dentro del propio ordenador (sin internet, sin
cuentas ni límites) para volver a pintar la zona:

1. un lector de imágenes (BLIP) mira la zona y dice qué hay («la cabeza de un
   gato blanco»): sin eso, la IA pinta una mancha;
2. el pintor la repinta partiendo de lo que queda (silueta, restos de ojos),
   acelerado con LCM (4 pasos en vez de 25);
3. se le devuelve el grano del lienzo del original y se funde con lo de
   alrededor.

Nunca se toca una cara humana: eso lo decide quien llama.

Instalación (una vez, ~3 GB): python setup_models.py --pintor
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from .faces import MODELS_DIR

# carpetas de los modelos (la primera que exista)
PAINTER_DIRS = ("sd_inpaint", "pintor")
LCM_DIR = "lcm_lora"
TINY_VAE_DIR = "taesd"
READER_DIR = "blip"
REPOS = {
    "sd_inpaint": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    LCM_DIR: "latent-consistency/lcm-lora-sdv1-5",
    READER_DIR: "Salesforce/blip-image-captioning-base",
    TINY_VAE_DIR: "madebyollin/taesd",
}
PROMPT = "old oil painting on canvas, {caption}, muted colors, soft brushstrokes, painterly"
GENERIC = "the same subject as around it"
NEGATIVE = "cracks, flaking, paint loss, damage, stains, blurry, text, watermark, frame, deformed, cartoon, sharp lines"
CAPTION_PREFIX = "the head of a"  # lo que se suele tocar: una cabeza perdida
SIDE = 384        # lado de trabajo (más rápido en CPU; 512 sin acelerador)
STRENGTH = 0.75   # cuánto se aleja de lo que hay (1 = de cero)
STEPS_LCM, GUIDANCE_LCM = 4, 1.5
STEPS, GUIDANCE = 25, 6.0
GRAIN = 0.6       # cuánto grano del lienzo original se devuelve

_pipe = None
_reader = None
_lock = threading.Lock()


def _dir(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def painter_dir() -> str | None:
    for name in PAINTER_DIRS:
        if os.path.exists(os.path.join(_dir(name), "model_index.json")):
            return _dir(name)
    return None


def available() -> bool:
    if painter_dir() is None:
        return False
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _fast() -> bool:
    return os.path.isdir(_dir(LCM_DIR)) and bool(os.listdir(_dir(LCM_DIR)))


def _load():
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import LCMScheduler, StableDiffusionInpaintPipeline

        path = painter_dir()
        fp16 = os.path.exists(os.path.join(path, "unet", "diffusion_pytorch_model.fp16.safetensors"))
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            path, variant="fp16" if fp16 else None, torch_dtype=torch.float32,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False, local_files_only=True,
        )
        if _fast():
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(_dir(LCM_DIR))
            pipe.fuse_lora()
        if os.path.exists(os.path.join(_dir(TINY_VAE_DIR), "config.json")):
            # codificador/decodificador pequeño: un 30 % más rápido en CPU, y
            # el detalle fino lo pone después el grano del lienzo original
            from diffusers import AutoencoderTiny

            pipe.vae = AutoencoderTiny.from_pretrained(_dir(TINY_VAE_DIR), torch_dtype=torch.float32, local_files_only=True)
        pipe.set_progress_bar_config(disable=True)
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        _pipe = pipe
    return _pipe


HUMAN = {"woman", "man", "girl", "boy", "lady", "person", "people", "child", "baby", "women", "men", "her", "his"}
ANIMALS = ("cat", "kitten", "dog", "puppy", "horse", "bird", "rabbit", "lamb", "sheep", "cow", "fox", "deer", "lion")
COLORS = (("white", lambda L, C, h: L > 175 and C < 22), ("black", lambda L, C, h: L < 60),
          ("grey", lambda L, C, h: C < 12), ("ginger", lambda L, C, h: 40 <= h <= 80 and C >= 22),
          ("brown", lambda L, C, h: True))


def _reader_models():
    global _reader
    if _reader is None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        _reader = (BlipProcessor.from_pretrained(_dir(READER_DIR), local_files_only=True),
                   BlipForConditionalGeneration.from_pretrained(_dir(READER_DIR), local_files_only=True).eval())
    return _reader


def _caption(crop: np.ndarray, prefix: str) -> str:
    from PIL import Image

    proc, model = _reader_models()
    inputs = proc(Image.fromarray(np.ascontiguousarray(crop)), prefix or None, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=20)
    text = proc.decode(out[0], skip_special_tokens=True).replace(" - ", "-").strip()
    return text[len(prefix):].strip() if prefix and text.startswith(prefix.replace(" - ", "-")) else text


def _tone(crop: np.ndarray, hole: np.ndarray) -> str:
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    px = lab[hole] if hole.any() else lab.reshape(-1, 3)
    L, a, b = np.median(px, axis=0)
    C = float(np.hypot(a - 128, b - 128))
    hue = float(np.degrees(np.arctan2(b - 128, a - 128)) % 360)
    return next(name for name, ok in COLORS if ok(L, C, hue))


def warm_up() -> None:
    """Carga el pintor y el lector antes de que se pidan (en un hilo aparte:
    la carga no bloquea al resto de REVO)."""
    try:
        with _lock:
            _load()
        if os.path.isdir(_dir(READER_DIR)):
            _reader_models()
    except Exception:  # noqa: BLE001  se volverá a intentar al usarlo
        pass


def describe(crop: np.ndarray, hole: np.ndarray | None = None) -> str:
    """Qué hay en la zona, en pocas palabras (inglés, para el pintor). Se lee
    el recorte justo de la zona (con más contexto, el lector ve a la persona
    de al lado y dice «una mujer»). Si es un animal, se pide su cabeza con su
    color («the face of a white cat»); nunca se pide pintar una persona."""
    if not os.path.isdir(_dir(READER_DIR)):
        return GENERIC
    try:
        words = []
        for prefix in ("a close-up painting of a", "the head of a"):
            words += _caption(crop, prefix).lower().replace(",", " ").split()
        animal = next((w for w in words if w.rstrip("s") in ANIMALS), None)
        if animal:
            tone = _tone(crop, hole if hole is not None else np.ones(crop.shape[:2], bool))
            return f"close-up of the face of a {tone} {animal.rstrip('s')}, eyes, nose, fur"
        # sin animal: nunca una cara ni una persona. Si es piel (un brazo,
        # un hombro), se pide piel; si no, lo mismo que hay alrededor
        if any(w in words for w in ("arm", "arms", "shoulder", "hand", "hands", "skin", "elbow")) or HUMAN & set(words):
            return "smooth bare skin of an arm, soft shading, no face"
        bad = HUMAN | {"face", "head", "'", "s", "is", "of", "a", "an", "the", "with", "in", "on", "and"}
        subject = " ".join(w for w in words[:10] if w not in bad)
        return subject or GENERIC
    except Exception:  # noqa: BLE001  sin lector se pinta igual, con menos guía
        return GENERIC


def _paint(crop: np.ndarray, hole: np.ndarray, seed: int) -> np.ndarray:
    import torch
    from PIL import Image

    h, w = crop.shape[:2]
    f = SIDE / max(h, w)
    sw, sh = max(64, int(round(w * f / 8)) * 8), max(64, int(round(h * f / 8)) * 8)
    ys, xs = np.nonzero(hole)
    tight = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)) if ys.size else (slice(None), slice(None))
    caption = describe(crop[tight], hole[tight])
    fast = _fast()
    img = Image.fromarray(cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA))
    msk = Image.fromarray(cv2.resize(hole.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST))
    with _lock:
        out = _load()(
            prompt=PROMPT.format(caption=caption), negative_prompt=None if fast else NEGATIVE,
            image=img, mask_image=msk, width=sw, height=sh, strength=STRENGTH,
            num_inference_steps=STEPS_LCM if fast else STEPS, guidance_scale=GUIDANCE_LCM if fast else GUIDANCE,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
    gen = cv2.resize(np.asarray(out.convert("RGB")), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # el grano del lienzo: la IA pinta liso, el cuadro tiene trama
    c = crop.astype(np.float32)
    gen += GRAIN * (c - cv2.GaussianBlur(c, (0, 0), 1.2))
    return np.clip(gen, 0, 255).astype(np.uint8)


def repaint(rgb: np.ndarray, zones: np.ndarray, say=None, paint=None) -> tuple[np.ndarray, int]:
    """Repinta cada zona (>0) en su recorte con margen y la funde con lo de
    alrededor. Devuelve (imagen, zonas repintadas)."""
    paint = paint or _paint
    out = rgb.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats((zones > 0).astype(np.uint8), connectivity=8)
    H, W = zones.shape
    done = 0
    for j in range(1, n):
        x, y, w, h = st[j, :4]
        p = max(32, int(0.35 * max(w, h)))  # contexto: el estilo de alrededor
        X0, Y0, X1, Y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
        hole = lab[Y0:Y1, X0:X1] == j
        if say:
            say(f"Repintando lo perdido ({j} de {n - 1})")
        rec = paint(np.ascontiguousarray(out[Y0:Y1, X0:X1]), hole, seed=j)
        # fundido suave: dentro de la zona, lo nuevo; en el borde, mezcla
        a = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 4)
        a = np.maximum(a, cv2.erode(hole.astype(np.uint8), np.ones((9, 9), np.uint8)))[..., None]
        out[Y0:Y1, X0:X1] = np.clip(out[Y0:Y1, X0:X1] * (1 - a) + rec * a, 0, 255).astype(np.uint8)
        done += 1
    return out, done
