"""ONFR REVO — interfaz web local.

    python app.py            → abre http://127.0.0.1:7860
"""
from __future__ import annotations

import base64
import os
import random
import tempfile

import cv2
import gradio as gr
import numpy as np

from revo import Settings, process
from revo import colorize as colorizer
from revo import restore
from revo.analysis import analyze
from revo.faces import FaceGuard, face_masks
from revo.inpaint import lama_available, preload
from revo.pipeline import MAX_WORK, intervention_map, summary_lines

MODE_BUTTONS = [
    ("mejorar", "Mejorar"),
    ("restaurar", "Restaurar"),
    ("mejorar_restaurar", "Mejorar + Restaurar"),
]
ACTION = {"mejorar": "MEJORAR", "restaurar": "RESTAURAR", "mejorar_restaurar": "MEJORAR + RESTAURAR"}
SLIDER_LABEL = {"mejorar": "Mejora", "restaurar": "Restauración", "mejorar_restaurar": "Intervención"}
MODE_HELP = {
    "mejorar": "Fotos actuales o de baja calidad: resolución, ruido, artefactos, nitidez y luz moderada.",
    "restaurar": "Fotos antiguas o deterioradas: polvo, manchas, arañazos, roturas, ruido y contraste. Mantiene la época y la resolución.",
    "mejorar_restaurar": "Proceso completo: restauración → recuperación de calidad → aumento de resolución.",
}
BRUSH = "#ff2bd6"
MARK_RGBA = (255, 43, 214, 150)

CSS = """
.gradio-container, body {background:
  radial-gradient(900px 520px at 0% 0%, rgba(124,58,237,.38), transparent 62%),
  radial-gradient(800px 480px at 100% 0%, rgba(6,182,212,.36), transparent 62%),
  radial-gradient(900px 600px at 100% 100%, rgba(249,115,22,.26), transparent 62%),
  radial-gradient(900px 600px at 0% 100%, rgba(219,39,119,.30), transparent 62%),
  #fbf7ff !important}
#revo-title {text-align:center; margin-top:8px}
#revo-title h1 {font-size:3.4rem; letter-spacing:.45rem; margin-bottom:4px; font-weight:900;
  background:linear-gradient(110deg, #7c3aed, #db2777, #f97316, #eab308, #10b981, #06b6d4, #7c3aed);
  -webkit-background-clip:text; background-clip:text; color:transparent;
  background-size:300% auto; animation: revo-shine 8s linear infinite;
  filter: drop-shadow(0 4px 14px rgba(219,39,119,.25))}
#revo-title p {margin:2px 0; font-weight:600; color:#6d28d9}
#revo-title em {color:#db2777}
@keyframes revo-shine {to {background-position:300% center}}

/* modos: cada uno con su color */
.mode-btn button, button.mode-btn {min-height:88px !important; font-size:1.2rem !important; color:#fff !important;
  border-radius:18px !important; font-weight:800 !important; border:none !important;
  transition:transform .15s, box-shadow .15s, filter .15s; filter:saturate(.75) brightness(1.05); opacity:.82}
button.mode-btn:hover {transform:translateY(-3px); opacity:1}
button.mode-btn.primary {filter:none; opacity:1; transform:translateY(-2px) scale(1.02);
  box-shadow:0 10px 28px rgba(219,39,119,.45), 0 0 0 3px #fff, 0 0 0 6px rgba(124,58,237,.55) !important}
button.mode-btn:nth-child(1) {background:linear-gradient(135deg,#06b6d4,#3b82f6) !important}
button.mode-btn:nth-child(2) {background:linear-gradient(135deg,#f59e0b,#f97316) !important}
button.mode-btn:nth-child(3) {background:linear-gradient(110deg, #7c3aed 0%, #db2777 45%, #f97316 75%, #06b6d4 100%) !important}
button.mode-btn:nth-child(4) {background:linear-gradient(135deg,#a3a3a3,#d4d4d4) !important; color:#fff !important}

/* paneles con brillo de color */
.gradio-container .block {border-radius:18px !important}
.gradio-container .gr-group, .gradio-container .form {border-radius:18px !important}
.gradio-container [data-testid="block-label"] {background:linear-gradient(110deg, #7c3aed 0%, #db2777 45%, #f97316 75%, #06b6d4 100%) !important; color:#fff !important;
  border:none !important; font-weight:700}
.gradio-container [data-testid="block-label"] svg {color:#fff !important}
.gradio-container [data-testid="block-info"] {color:#7c3aed !important; font-weight:700}
.gradio-container input[type=range] {accent-color:#db2777}

#go-btn {min-height:64px; font-size:1.3rem; letter-spacing:.14rem; font-weight:900; border:none !important;
  color:#fff !important; border-radius:18px !important;
  background:linear-gradient(110deg, #7c3aed, #db2777, #f97316, #db2777, #7c3aed) !important;
  background-size:300% auto !important; box-shadow:0 12px 30px rgba(219,39,119,.45);
  animation: revo-shine 6s linear infinite}
#go-btn:disabled {filter:grayscale(.5); opacity:.7}
#lock {border-left:5px solid #10b981; padding:6px 12px; border-radius:10px;
  background:linear-gradient(90deg, rgba(16,185,129,.18), rgba(6,182,212,.08))}
#save-btn {background:linear-gradient(110deg,#10b981,#06b6d4) !important; color:#fff !important; border:none !important;
  font-weight:800 !important; box-shadow:0 8px 22px rgba(16,185,129,.4)}
#summary-btn {background:linear-gradient(110deg,#f59e0b,#ec4899) !important; color:#fff !important; border:none !important;
  font-weight:800 !important; box-shadow:0 8px 22px rgba(236,72,153,.35)}

/* ---- animación de procesado (destellos, estilo borrador IA) ---- */
.revo-anim {position:relative; overflow:hidden; border-radius:14px; background:#111;
  display:flex; justify-content:center; align-items:center; min-height:320px}
.revo-anim img {max-width:100%; max-height:560px; display:block; filter:saturate(.85) brightness(.92);
  animation: revo-breathe 2.4s ease-in-out infinite}
.revo-anim .glow {position:absolute; inset:0; pointer-events:none; border-radius:14px;
  box-shadow: inset 0 0 40px 6px rgba(140,110,255,.55), inset 0 0 90px 10px rgba(80,200,255,.25);
  animation: revo-glow 2.4s ease-in-out infinite}
.revo-anim .sweep {position:absolute; top:-50%; bottom:-50%; width:45%; left:-60%; pointer-events:none;
  background: linear-gradient(100deg, transparent 0%, rgba(120,170,255,.0) 20%, rgba(170,140,255,.35) 42%,
  rgba(255,255,255,.75) 50%, rgba(120,220,255,.35) 58%, transparent 80%);
  mix-blend-mode: screen; filter: blur(6px); transform: rotate(12deg);
  animation: revo-sweep 1.9s cubic-bezier(.45,.05,.35,1) infinite}
.revo-anim .spark {position:absolute; width:var(--s); height:var(--s); border-radius:50%; pointer-events:none;
  background: radial-gradient(circle, #fff 0%, rgba(200,180,255,.9) 30%, rgba(120,200,255,0) 70%);
  opacity:0; animation: revo-twinkle var(--d) ease-in-out var(--t) infinite}
.revo-anim .spark::before, .revo-anim .spark::after {content:""; position:absolute; left:50%; top:50%;
  width:300%; height:2px; background:linear-gradient(90deg,transparent,#fff,transparent);
  transform:translate(-50%,-50%)}
.revo-anim .spark::after {transform:translate(-50%,-50%) rotate(90deg)}
@keyframes revo-sweep {0% {left:-60%} 100% {left:120%}}
@keyframes revo-twinkle {0%,100% {opacity:0; transform:scale(.3)} 45% {opacity:1; transform:scale(1)} 60% {opacity:.8}}
@keyframes revo-glow {0%,100% {opacity:.55} 50% {opacity:1}}
@keyframes revo-breathe {0%,100% {filter:saturate(.85) brightness(.92)} 50% {filter:saturate(1) brightness(1.02)}}
"""

# carga los modelos de caras al arrancar, no en la primera foto
FaceGuard.get()
preload()  # LaMa se carga mientras eliges la foto
if colorizer.available():  # y el modelo de color también
    import threading

    threading.Thread(target=colorizer._load, daemon=True).start()


def mode_updates(mode):
    btns = [gr.update(variant="primary" if m == mode else "secondary") for m, _ in MODE_BUTTONS]
    return (
        mode,
        *btns,
        gr.update(label=f"{SLIDER_LABEL[mode]}  ·  Conservadora ⟷ Intensa"),
        gr.update(value=ACTION[mode]),
        f"_{MODE_HELP[mode]}_",
        gr.update(visible=mode != "restaurar"),
    )


def _background(ed):
    if ed is None:
        return None
    bg = ed.get("background") if isinstance(ed, dict) else ed
    if bg is None:
        return None
    bg = np.asarray(bg)
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(bg[..., :3])


def _painted(ed, shape):
    """Máscara de lo pintado con el pincel (todas las capas)."""
    mask = np.zeros(shape[:2], np.uint8)
    if not isinstance(ed, dict):
        return mask
    for layer in ed.get("layers") or []:
        layer = np.asarray(layer)
        if layer.ndim == 3 and layer.shape[2] == 4:
            a = layer[..., 3]
            if a.shape != mask.shape:
                a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            mask |= (a > 0).astype(np.uint8)
    return mask


def on_upload(ed, mode, user_picked, intensity):
    img = _background(ed)
    if img is None:
        return gr.update(), "", gr.update(visible=False), mode, *mode_updates(mode)[1:]
    big = max(img.shape[:2]) > MAX_WORK
    if big:  # fotos de móvil enormes: se trabaja a MAX_WORK px (mucho más rápido)
        f = MAX_WORK / max(img.shape[:2])
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    a = analyze(img)
    text = "**REVO ha analizado la imagen**\n\n" + "\n".join(f"- {d}" for d in a.describe())
    if big:
        text += f"\n- Foto muy grande: se trabaja a {img.shape[1]} × {img.shape[0]} px para ir rápido"
    if not user_picked:
        mode = a.suggested_mode()
        text += f"\n\nModo sugerido: **{dict(MODE_BUTTONS)[mode]}**"

    # daños grandes sugeridos: se pintan en la capa del pincel para que puedas
    # revisarlos (borrar o añadir) antes de reparar
    layer = np.zeros((*img.shape[:2], 4), np.uint8)
    if a.monochrome:
        faces = FaceGuard.get().detect(img)
        fu = np.zeros(img.shape[:2], np.float32)
        ft = np.zeros(img.shape[:2], np.float32)
        for f in faces:
            m1, m2 = face_masks(img.shape, f)
            fu, ft = np.maximum(fu, m1), np.maximum(ft, m2)
        dmg, info = restore.detect_damage(img, True, fu, ft, intensity / 100)
        if info["regions"]:
            layer[dmg > 0] = MARK_RGBA
            text += (
                f"\n\nHe marcado en rosa **{info['regions']} daños** (roturas, marcas). "
                "Revísalos con el pincel o la goma antes de reparar."
            )
    if not a.monochrome or not layer[..., 3].any():
        text += "\n\nSi ves agujeros o rasguños, píntalos con el pincel y REVO los rellenará."
    new_value = {"background": img, "layers": [layer], "composite": None}
    return new_value, text, gr.update(visible=a.monochrome), mode, *mode_updates(mode)[1:]


def anim_html(ed):
    img = _background(ed)
    if img is None:
        return gr.update(visible=False), gr.update()
    h, w = img.shape[:2]
    f = min(1.0, 900 / max(h, w))
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode()
    rnd = random.Random(7)
    sparks = "".join(
        f'<span class="spark" style="left:{rnd.uniform(4, 96):.1f}%;top:{rnd.uniform(4, 96):.1f}%;'
        f'--s:{rnd.uniform(5, 13):.0f}px;--d:{rnd.uniform(1.4, 2.6):.2f}s;--t:{rnd.uniform(0, 2.4):.2f}s"></span>'
        for _ in range(38)
    )
    html = (
        f'<div class="revo-anim"><img src="data:image/jpeg;base64,{b64}"/>'
        f'<div class="sweep"></div>{sparks}<div class="glow"></div></div>'
    )
    return gr.update(value=html, visible=True), gr.update(visible=False)


def run(ed, mode, intensity, scale, color_on, color_amount):
    img = _background(ed)
    if img is None:
        raise gr.Error("Primero sube una imagen.")
    s = Settings(
        mode=mode,
        intensity=intensity / 100.0,
        scale=int(scale.rstrip("×")),
        colorize=bool(color_on),
        color_amount=color_amount / 100.0,
        user_mask=_painted(ed, img.shape),
        auto_damage=False,  # los daños grandes ya están en la capa del pincel
    )
    try:
        res = process(img, s)
    except Exception as e:  # noqa: BLE001  la animación no debe quedarse girando
        import traceback

        traceback.print_exc()
        return (
            None,
            gr.update(visible=True),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=f"**No se ha podido procesar la imagen:** {e}", visible=True),
            gr.update(value="", visible=False),
            gr.update(value=None, visible=False),
        )
    out_path = os.path.join(tempfile.mkdtemp(prefix="revo_"), "revo_resultado.png")
    cv2.imwrite(out_path, cv2.cvtColor(res.image, cv2.COLOR_RGB2BGR))
    t = res.stats["seconds"]
    return (
        res,
        gr.update(value=(res.before, res.image), visible=True),
        gr.update(value="", visible=False),
        gr.update(value=out_path, visible=True),
        gr.update(visible=True),
        gr.update(value=f"_Listo en {t:.1f} s._", visible=True),
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
    )


def show_summary(res):
    if res is None:
        return gr.update(), gr.update()
    imap, changed = intervention_map(res)
    return gr.update(value="\n".join(summary_lines(res, changed)), visible=True), gr.update(value=imap, visible=True)


COLOR_OK = colorizer.available()

with gr.Blocks(title="ONFR REVO") as demo:
    gr.Markdown(
        "# ONFR REVO\nRestore without altering.\n\n_Mejora la imagen. Conserva el original._",
        elem_id="revo-title",
    )
    mode = gr.State("mejorar_restaurar")
    user_picked = gr.State(False)
    result = gr.State(None)

    with gr.Row():
        buttons = [
            gr.Button(label, variant="primary" if m == "mejorar_restaurar" else "secondary", elem_classes="mode-btn")
            for m, label in MODE_BUTTONS
        ]
        gr.Button("Restaurar cuadro\n(próximamente)", interactive=False, elem_classes="mode-btn")
    mode_help = gr.Markdown(f"_{MODE_HELP['mejorar_restaurar']}_")

    with gr.Row():
        with gr.Column(scale=1):
            editor = gr.ImageEditor(
                label="Sube tu imagen · pinta encima de agujeros y rasguños",
                type="numpy",
                sources=["upload", "clipboard"],
                brush=gr.Brush(colors=[BRUSH], default_color=BRUSH, color_mode="fixed", default_size=18),
                eraser=gr.Eraser(default_size=24),
                transforms=(),
                layers=False,
                height=520,
            )
            analysis_md = gr.Markdown()
            intensity = gr.Slider(0, 100, value=35, step=1, label="Intervención  ·  Conservadora ⟷ Intensa")
            with gr.Group(visible=False) as color_box:
                color_on = gr.Checkbox(
                    label=(
                        "Colorear  (los colores son una estimación de la IA, no información recuperada)"
                        if COLOR_OK
                        else "Colorear  (falta el modelo de color: ejecuta setup_models.py)"
                    ),
                    value=False,
                    interactive=COLOR_OK,
                )
                color_amount = gr.Slider(0, 100, value=80, step=1, label="Intensidad del color", visible=False)
            gr.Markdown(
                "**Protección facial  ● ON** — siempre activa. REVO no inventa rasgos: si no puede "
                "mejorar una cara con seguridad, la deja como estaba.",
                elem_id="lock",
            )
            with gr.Accordion("Ajustes avanzados", open=False):
                with gr.Group() as scale_box:
                    scale = gr.Radio(["2×", "4×", "1×"], value="2×", label="Aumento de resolución")
            go = gr.Button(ACTION["mejorar_restaurar"], variant="primary", elem_id="go-btn")

        with gr.Column(scale=1):
            anim = gr.HTML(visible=False)
            slider = gr.ImageSlider(label="ANTES ⟷ DESPUÉS", type="numpy", format="png", max_height=560)
            done_md = gr.Markdown(visible=False)
            with gr.Row():
                save = gr.DownloadButton("Guardar", visible=False, variant="primary", elem_id="save-btn")
                summary_btn = gr.Button("Ver resumen", visible=False, elem_id="summary-btn")
            summary = gr.Markdown(visible=False)
            imap = gr.Image(label="Mapa de intervención", type="numpy", interactive=False, visible=False)
            gr.Markdown(
                "<small>Mapa: calor = retoques locales · magenta = daños reparados automáticamente · "
                "cian = zonas marcadas con el pincel · recuadro verde = cara mejorada y verificada · naranja = cara conservada.</small>",
                visible=True,
            )

    mode_outputs = [mode, *buttons, intensity, go, mode_help, scale_box]
    for (m, _), b in zip(MODE_BUTTONS, buttons):
        b.click(lambda m=m: mode_updates(m), None, mode_outputs).then(lambda: True, None, user_picked)

    # el botón espera a que termine el análisis (y las marcas rosas)
    editor.upload(lambda: gr.update(interactive=False, value="Analizando la imagen…"), None, go).then(
        on_upload, [editor, mode, user_picked, intensity], [editor, analysis_md, color_box, *mode_outputs]
    ).then(lambda m: gr.update(interactive=True, value=ACTION[m]), mode, go)
    color_on.change(lambda on: gr.update(visible=bool(on)), color_on, color_amount)
    go.click(anim_html, editor, [anim, slider], show_progress="hidden").then(
        run,
        [editor, mode, intensity, scale, color_on, color_amount],
        [result, slider, anim, save, summary_btn, done_md, summary, imap],
        show_progress="hidden",
    )
    summary_btn.click(show_summary, result, [summary, imap])


if __name__ == "__main__":
    demo.queue().launch(
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="pink", neutral_hue="slate"),
        css=CSS,
        server_name=os.environ.get("REVO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("REVO_PORT", "7860")),
        inbrowser=os.environ.get("REVO_OPEN_BROWSER") == "1",
    )
