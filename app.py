"""ONFR REVO — interfaz web local.

    python app.py            → abre http://127.0.0.1:7860
"""
from __future__ import annotations

import os
import tempfile

import cv2
import gradio as gr
import numpy as np

from revo import Settings, process, summary_lines
from revo.analysis import analyze

MODE_BUTTONS = [
    ("mejorar", "Mejorar"),
    ("restaurar", "Restaurar"),
    ("mejorar_restaurar", "Mejorar + Restaurar"),
]
ACTION = {"mejorar": "MEJORAR", "restaurar": "RESTAURAR", "mejorar_restaurar": "MEJORAR + RESTAURAR"}
SLIDER_LABEL = {"mejorar": "Mejora", "restaurar": "Restauración", "mejorar_restaurar": "Intervención"}
MODE_HELP = {
    "mejorar": "Fotos actuales o de baja calidad: resolución, ruido, artefactos, nitidez y luz moderada.",
    "restaurar": "Fotos antiguas o deterioradas: polvo, manchas, arañazos finos, ruido y contraste. Mantiene la época y la resolución.",
    "mejorar_restaurar": "Proceso completo: restauración → recuperación de calidad → aumento de resolución.",
}

CSS = """
#revo-title {text-align:center; margin-top:8px}
#revo-title h1 {font-size:2.6rem; letter-spacing:.35rem; margin-bottom:0}
#revo-title p {margin:2px 0; opacity:.75}
.mode-btn button, button.mode-btn {min-height:84px !important; font-size:1.15rem !important}
#go-btn {min-height:56px; font-size:1.2rem; letter-spacing:.1rem}
#lock {opacity:.85}
"""


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


def on_upload(img, mode, user_picked):
    if img is None:
        return "", gr.update(visible=False), mode, *mode_updates(mode)[1:]
    a = analyze(img[..., :3])
    text = "**REVO ha analizado la imagen**\n\n" + "\n".join(f"- {d}" for d in a.describe())
    if not user_picked:
        mode = a.suggested_mode()
        text += f"\n\nModo sugerido: **{dict(MODE_BUTTONS)[mode]}**"
    return text, gr.update(visible=a.monochrome), mode, *mode_updates(mode)[1:]


def run(img, mode, intensity, scale, progress=gr.Progress()):
    if img is None:
        raise gr.Error("Primero sube una imagen.")
    s = Settings(mode=mode, intensity=intensity / 100.0, scale=int(scale.rstrip("×")))
    res = process(img[..., :3], s, progress=lambda f, m: progress(f, desc=m))
    out_path = os.path.join(tempfile.mkdtemp(prefix="revo_"), "revo_resultado.png")
    cv2.imwrite(out_path, cv2.cvtColor(res.image, cv2.COLOR_RGB2BGR))
    return (
        (res.before, res.image),
        "\n".join(summary_lines(res)),
        res.intervention_map,
        gr.update(value=out_path, visible=True),
    )


with gr.Blocks(title="ONFR REVO") as demo:
    gr.Markdown(
        "# ONFR REVO\nRestore without altering.\n\n_Mejora la imagen. Conserva el original._",
        elem_id="revo-title",
    )
    mode = gr.State("mejorar_restaurar")
    user_picked = gr.State(False)

    with gr.Row():
        buttons = [
            gr.Button(label, variant="primary" if m == "mejorar_restaurar" else "secondary", elem_classes="mode-btn")
            for m, label in MODE_BUTTONS
        ]
        gr.Button("Restaurar cuadro\n(próximamente)", interactive=False, elem_classes="mode-btn")
    mode_help = gr.Markdown(f"_{MODE_HELP['mejorar_restaurar']}_")

    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(label="Sube tu imagen", type="numpy", sources=["upload", "clipboard"], height=360)
            analysis_md = gr.Markdown()
            intensity = gr.Slider(
                0, 100, value=35, step=1, label="Intervención  ·  Conservadora ⟷ Intensa"
            )
            with gr.Group(visible=False) as color_box:
                gr.Checkbox(
                    label="Colorización  ○ OFF  (próximamente: será opcional y los colores serán una estimación)",
                    value=False,
                    interactive=False,
                )
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
            slider = gr.ImageSlider(label="ANTES ⟷ DESPUÉS", type="numpy", format="png", max_height=560)
            summary = gr.Markdown()
            save = gr.DownloadButton("Guardar", visible=False, variant="primary")
            with gr.Accordion("Mapa de intervención (qué se ha tocado)", open=False):
                gr.Markdown(
                    "Calor = retoques locales (ruido, nitidez, contraste local). "
                    "Magenta = defectos rellenados. Recuadro verde = cara mejorada y verificada; "
                    "naranja = cara conservada como el original."
                )
                imap = gr.Image(show_label=False, type="numpy", interactive=False)

    mode_outputs = [mode, *buttons, intensity, go, mode_help, scale_box]
    for (m, _), b in zip(MODE_BUTTONS, buttons):
        b.click(lambda m=m: mode_updates(m), None, mode_outputs).then(lambda: True, None, user_picked)

    img.upload(on_upload, [img, mode, user_picked], [analysis_md, color_box, *mode_outputs])
    go.click(run, [img, mode, intensity, scale], [slider, summary, imap, save])


if __name__ == "__main__":
    demo.queue().launch(
        theme=gr.themes.Soft(primary_hue="neutral"),
        css=CSS,
        server_name=os.environ.get("REVO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("REVO_PORT", "7860")),
        inbrowser=os.environ.get("REVO_OPEN_BROWSER") == "1",
    )
