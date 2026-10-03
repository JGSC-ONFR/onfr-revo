"""Repintado de lo que se ha perdido del todo (IA generativa local).

Donde un cuadro ha perdido la pintura por completo (la cara de un gato de la
que solo quedan dos manchitas de los ojos), restaurar no basta: no hay nada
debajo que recuperar. Ahí, y solo ahí, REVO usa Stable Diffusion Inpainting
dentro del propio ordenador (sin internet, sin cuentas ni límites) para volver
a pintar la zona con el estilo y los colores de alrededor. Parte de lo que
queda (silueta, restos de ojos), no de cero.

Nunca se toca una cara humana: eso lo decide quien llama.

Instalación (una vez, ~2 GB): python setup_models.py --pintor
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from .faces import MODELS_DIR

MODEL_DIR = os.path.join(MODELS_DIR, "pintor")
REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
PROMPT = (
    "restored oil painting, intact paint, same subject, same style and colors as the rest of the painting, "
    "fine brushwork, detailed, masterpiece"
)
NEGATIVE = "cracks, flaking, paint loss, damage, stains, blurry, smudge, text, watermark, frame, deformed"
SIDE = 512        # SD 1.5 trabaja a 512 px
STEPS = 25        # en CPU: ~1-3 min por zona
STRENGTH = 0.85   # cuánto se aleja de lo que hay (1 = de cero)
GUIDANCE = 6.0

_pipe = None
_lock = threading.Lock()


def available() -> bool:
    if not os.path.exists(os.path.join(MODEL_DIR, "model_index.json")):
        return False
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _load():
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        fp16 = os.path.exists(os.path.join(MODEL_DIR, "unet", "diffusion_pytorch_model.fp16.safetensors"))
        _pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_DIR, variant="fp16" if fp16 else None, torch_dtype=torch.float32,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False, local_files_only=True,
        )
        _pipe.set_progress_bar_config(disable=True)
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    return _pipe


def _paint(crop: np.ndarray, hole: np.ndarray, seed: int) -> np.ndarray:
    import torch
    from PIL import Image

    h, w = crop.shape[:2]
    f = SIDE / max(h, w)
    sw, sh = max(64, int(round(w * f / 8)) * 8), max(64, int(round(h * f / 8)) * 8)
    img = Image.fromarray(cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA))
    msk = Image.fromarray(cv2.resize(hole.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST))
    with _lock:
        out = _load()(
            prompt=PROMPT, negative_prompt=NEGATIVE, image=img, mask_image=msk,
            width=sw, height=sh, strength=STRENGTH, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
    return cv2.resize(np.asarray(out.convert("RGB")), (w, h), interpolation=cv2.INTER_CUBIC)


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
        r = max(2, int(0.02 * max(w, h)))
        a = cv2.GaussianBlur(cv2.dilate(hole.astype(np.uint8), np.ones((r, r), np.uint8)).astype(np.float32), (0, 0), r / 2)
        a = np.maximum(a, hole)[..., None]
        out[Y0:Y1, X0:X1] = np.clip(out[Y0:Y1, X0:X1] * (1 - a) + rec * a, 0, 255).astype(np.uint8)
        done += 1
    return out, done
