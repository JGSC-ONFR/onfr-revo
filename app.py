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
.gradio-container {background:
  radial-gradient(1200px 600px at 0% -10%, rgba(124,58,237,.22), transparent 60%),
  radial-gradient(900px 500px at 100% 0%, rgba(6,182,212,.20), transparent 60%),
  radial-gradient(900px 600px at 50% 110%, rgba(219,39,119,.16), transparent 60%) !important}
#revo-title {text-align:center; margin-top:8px}
#revo-title h1 {font-size:3rem; letter-spacing:.4rem; margin-bottom:0; font-weight:800;
  background:linear-gradient(110deg, #7c3aed 0%, #db2777 45%, #f97316 75%, #06b6d4 100%); -webkit-background-clip:text; background-clip:text; color:transparent;
  background-size:200% auto; animation: revo-shine 6s linear infinite}
#revo-title p {margin:2px 0; opacity:.8}
@keyframes revo-shine {to {background-position:200% center}}
.mode-btn button, button.mode-btn {min-height:84px !important; font-size:1.15rem !important;
  border-radius:16px !important; font-weight:700 !important; transition:transform .15s, box-shadow .15s}
button.mode-btn:hover {transform:translateY(-2px)}
button.mode-btn.primary {background:linear-gradient(110deg, #7c3aed 0%, #db2777 45%, #f97316 75%, #06b6d4 100%) !important; color:#fff !important; border:none !important;
  box-shadow:0 8px 24px rgba(219,39,119,.35)}
button.mode-btn.secondary {border:2px solid rgba(124,58,237,.35) !important}
#go-btn {min-height:60px; font-size:1.25rem; letter-spacing:.12rem; font-weight:800; border:none !important;
  color:#fff !important; border-radius:16px !important; background:linear-gradient(110deg, #7c3aed 0%, #db2777 45%, #f97316 75%, #06b6d4 100%) !important;
  background-size:200% auto !important; box-shadow:0 10px 28px rgba(124,58,237,.4);
  animation: revo-shine 5s linear infinite}
#lock {border-left:4px solid #10b981; padding-left:10px; border-radius:6px; background:rgba(16,185,129,.08)}
#save-btn {background:linear-gradient(110deg,#10b981,#06b6d4) !important; color:#fff !important; border:none !important}
#summary-btn {border:2px solid rgba(6,182,212,.6) !important}
.block, .form {border-radius:16px !important}

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

    editor.upload(
        on_upload, [editor, mode, user_picked, intensity], [editor, analysis_md, color_box, *mode_outputs]
    )
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
