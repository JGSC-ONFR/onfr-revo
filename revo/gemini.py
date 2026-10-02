"""Restauración de cuadros con Gemini (IA generativa de Google, en la nube).

La foto se envía a Google con instrucciones de restauración conservadora y
se recibe el cuadro restaurado. A diferencia del restaurador propio de REVO,
la IA vuelve a pintar las zonas perdidas: puede cambiar algún detalle, y así
se avisa en el resumen.

La clave se guarda solo en este ordenador: variable de entorno
GEMINI_API_KEY o el archivo revo_config.json junto a la app (no se sube a
GitHub).
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "revo_config.json")
KEY_FILE = os.path.join(ROOT, "revo_clave.txt")  # también se puede pegar la clave aquí, a mano
# de mejor a peor; si uno no está disponible para la clave, se prueba el siguiente
MODELS = ("gemini-2.5-flash-image", "gemini-2.5-flash-image-preview")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SEND_SIDE = 2048  # px del lado largo que se envían

PROMPT = (
    "Restore this photograph of a damaged painting, like a professional conservator would. "
    "Remove all paint losses and flaking (where the white or cream ground shows through), cracks, "
    "scratches, stains, dirt, dust and yellowed varnish, and fill every missing area so it matches the "
    "surrounding paint seamlessly. "
    "Keep exactly the same composition, framing and proportions; the same faces, eyes, mouths and facial "
    "expressions; the same animals, clothes, objects, background, signature and date; the same colours, "
    "brushwork, painting style and canvas texture. "
    "Do not add, remove, redraw or embellish anything, do not modernise it and do not make it look "
    "newly painted. Return only the restored painting."
)


class GeminiError(RuntimeError):
    pass


def get_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        with open(CONFIG, encoding="utf-8") as f:
            key = str(json.load(f).get("gemini_api_key", "")).strip()
    except (OSError, ValueError):
        key = ""
    if not key:
        try:
            with open(KEY_FILE, encoding="utf-8-sig") as f:
                key = f.read().strip()
        except OSError:
            pass
    return key


def save_key(key: str) -> None:
    data = {}
    try:
        with open(CONFIG, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        pass
    data["gemini_api_key"] = key.strip()
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _post(model: str, key: str, body: bytes) -> dict:
    req = urllib.request.Request(
        ENDPOINT.format(model=model),
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", {}).get("message", "")
        except ValueError:
            msg = ""
        raise GeminiError(f"{e.code}: {msg}") from None
    except urllib.error.URLError as e:
        raise GeminiError(f"sin conexión con Google ({e.reason})") from None


def _explain(err: str) -> str:
    if "API key not valid" in err or "API_KEY_INVALID" in err:
        return "la clave de Gemini no es válida. Revísala en Ajustes avanzados."
    if err.startswith("429"):
        return "Gemini dice que se ha superado el límite de uso de tu clave. Espera un poco o revisa tu plan."
    if err.startswith("403"):
        return "tu clave de Gemini no tiene permiso para el modelo de imagen."
    return f"Gemini no ha podido restaurarla ({err})."


def restore(rgb: np.ndarray, key: str | None = None) -> tuple[np.ndarray, str]:
    """Devuelve (cuadro restaurado al mismo tamaño que `rgb`, modelo usado)."""
    key = key or get_key()
    if not key:
        raise GeminiError("falta la clave de Gemini: pégala en Ajustes avanzados.")
    h, w = rgb.shape[:2]
    f = min(1.0, SEND_SIDE / max(h, w))
    small = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    body = json.dumps({
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.tobytes()).decode()}},
        ]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }).encode()
    last = ""
    for model in MODELS:
        try:
            data = _post(model, key, body)
        except GeminiError as e:
            last = str(e)
            if last.startswith("404"):
                continue  # este modelo no existe para la clave: el siguiente
            raise GeminiError(_explain(last)) from None
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    raw = np.frombuffer(base64.b64decode(inline["data"]), np.uint8)
                    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if img is not None:
                        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if out.shape[:2] != (h, w):
                            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
                        return out, model
        reason = data.get("promptFeedback", {}).get("blockReason") or (
            (data.get("candidates") or [{}])[0].get("finishReason", "sin imagen")
        )
        raise GeminiError(f"Gemini no ha devuelto ninguna imagen ({reason}).")
    raise GeminiError(_explain(last or "404"))
