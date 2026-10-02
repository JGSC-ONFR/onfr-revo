"""Restauración de cuadros con IA gratuita en Hugging Face.

Orden: FLUX.1 Kontext → Qwen Image Edit. Si las dos fallan (sin internet,
límite diario agotado, servidor ocupado), quien llama restaura con REVO.
No hace falta cuenta ni clave; con un token gratuito de Hugging Face (variable
HF_TOKEN, "hf_token" en revo_config.json o revo_token_hf.txt, ninguno se sube
a GitHub) el límite diario es mayor.

El cuadro se envía a servidores públicos de Hugging Face.
"""
from __future__ import annotations

import json
import os
import tempfile

import cv2
import numpy as np

from .gemini import CONFIG, ROOT

TOKEN_FILE = os.path.join(ROOT, "revo_token_hf.txt")
PROMPT = (
    "Restore this damaged oil painting: remove the cracks, paint losses, flaking, stains and "
    "yellow varnish. Keep the exact same composition, faces, expressions, colors, brushwork and "
    "style. Do not add anything."
)
# (nombre que ve Jorge, espacios de Hugging Face de mejor a peor)
ENGINES = (
    ("FLUX", ("black-forest-labs/FLUX.1-Kontext-Dev",)),
    ("Qwen", ("Qwen/Qwen-Image-Edit", "multimodalart/Qwen-Image-Edit-Fast", "Qwen/Qwen-Image-Edit-2509")),
)
# Qwen pide más tiempo de GPU del que Hugging Face da sin cuenta (probado)
NEEDS_TOKEN = {"Qwen"}
SEND_SIDE = 1024  # estas IA trabajan a ~1 megapíxel
WAIT = 150  # s como mucho por espacio (cola + cálculo)
# valores fijos para que el resultado sea repetible y fiel
FIXED = {"seed": 0, "randomize_seed": False, "rewrite_prompt": False, "num_images_per_prompt": 1}


class CloudError(RuntimeError):
    pass


def _config() -> dict:
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def get_token() -> str | None:
    tok = os.environ.get("HF_TOKEN", "").strip() or str(_config().get("hf_token", "")).strip()
    if not tok:
        try:
            with open(TOKEN_FILE, encoding="utf-8-sig") as f:
                tok = f.read().strip()
        except OSError:
            pass
    return tok or None


def save_token(tok: str) -> None:
    data = _config()
    data["hf_token"] = tok.strip()
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f)


def check_token(tok: str) -> str:
    """"" si el token vale; si no, el motivo en palabras sencillas."""
    import urllib.error
    import urllib.request

    tok = (tok or "").strip()
    if not tok:
        return "no has puesto ningún token."
    if not tok.startswith("hf_"):
        return "los tokens de Hugging Face empiezan por «hf_»."
    req = urllib.request.Request("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()
        return ""
    except urllib.error.HTTPError as e:
        return "el token no es válido." if e.code == 401 else f"Hugging Face ha respondido con un error ({e.code})."
    except urllib.error.URLError as e:
        return f"no hay conexión con Hugging Face ({e.reason})."


def get_enabled() -> bool:
    return bool(_config().get("ia_on", True))


def set_enabled(on: bool) -> None:
    data = _config()
    data["ia_on"] = bool(on)
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _explain(err: str) -> str:
    low = err.lower()
    if "quota" in low or "exceeded" in low or "limit" in low:
        return "se ha agotado el uso gratuito de hoy"
    if "timeout" in low or "timed out" in low or "queue" in low or "busy" in low:
        return "está muy ocupada ahora mismo"
    if "connect" in low or "resolve" in low or "network" in low or "ssl" in low:
        return "no hay conexión con Hugging Face"
    if "paused" in low or "sleeping" in low or "runtime" in low or "not found" in low or "404" in low:
        return "no está disponible ahora mismo"
    return f"ha fallado ({err[:120]})"


def _find_image(res):
    """Busca la ruta de la imagen en lo que devuelva el espacio: una ruta,
    {"path": …}, una galería [{"image": …}], o una tupla (imagen, semilla)."""
    if isinstance(res, str):
        return res if os.path.exists(res) else None
    if isinstance(res, dict):
        for k in ("path", "image", "url", "value"):
            if k in res:
                p = _find_image(res[k])
                if p:
                    return p
        return None
    if isinstance(res, (list, tuple)):
        for it in res:
            p = _find_image(it)
            if p:
                return p
    return None


def _arguments(client, src: str) -> tuple[str, dict]:
    """Rellena los parámetros del espacio por su nombre: la imagen, el texto
    y los valores fijos; el resto se queda con lo que el espacio trae."""
    from gradio_client import handle_file

    api = client.view_api(print_info=False, return_format="dict")["named_endpoints"]
    name = "/infer" if "/infer" in api else next(
        (n for n, e in api.items() if any(p.get("component") in ("Image", "Gallery") for p in e["parameters"])), None)
    if name is None:
        raise CloudError("no ofrece edición de imágenes")
    kwargs, has_image = {}, False
    for p in api[name]["parameters"]:
        pn, comp = p.get("parameter_name") or "", p.get("component")
        if comp == "Image" and not has_image:
            kwargs[pn], has_image = handle_file(src), True
        elif comp == "Gallery" and not has_image:
            kwargs[pn], has_image = [{"image": handle_file(src), "caption": None}], True
        elif comp == "Textbox" and "prompt" in pn.lower() and "negative" not in pn.lower():
            kwargs[pn] = PROMPT
        elif pn in FIXED:
            kwargs[pn] = FIXED[pn]
    if not has_image:
        raise CloudError("no acepta imágenes")
    return name, kwargs


def _call(space: str, src: str) -> np.ndarray:
    from gradio_client import Client

    client = Client(space, token=get_token(), verbose=False)
    name, kwargs = _arguments(client, src)
    res = client.submit(api_name=name, **kwargs).result(timeout=WAIT)
    path = _find_image(res)
    out = cv2.imread(path, cv2.IMREAD_COLOR) if path else None
    if out is None:
        raise CloudError("no ha devuelto ninguna imagen")
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def restore(rgb: np.ndarray, engine: str, call=None) -> np.ndarray:
    """Restaura con una de ENGINES; devuelve el cuadro al tamaño del original
    o lanza CloudError con el motivo en palabras sencillas."""
    try:
        import gradio_client  # noqa: F401  viene con gradio
    except ImportError as e:
        raise CloudError("falta gradio_client") from e
    if call is None and engine in NEEDS_TOKEN and not get_token():
        raise CloudError("necesita un token gratuito de Hugging Face")
    call = call or _call
    spaces = dict(ENGINES)[engine]
    h, w = rgb.shape[:2]
    f = min(1.0, SEND_SIDE / max(h, w))
    small = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else rgb
    last = ""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "cuadro.png")
        cv2.imwrite(src, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        for space in spaces:
            try:
                out = call(space, src)
            except CloudError as e:
                last = str(e)
                continue
            except Exception as e:  # noqa: BLE001  cualquier fallo del servicio → siguiente
                last = _explain(f"{type(e).__name__}: {e}")
                continue
            return cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
    raise CloudError(last or "no ha respondido")
