"""Uso por línea de comandos:

    python -m revo foto.jpg                       # Mejorar + Restaurar, conservadora
    python -m revo foto.jpg -m restaurar -i 60 -o salida.png --mapa mapa.png
"""
import argparse
import os

import cv2

from . import MODES, Settings, process, summary_lines
from .pipeline import intervention_map


def main():
    ap = argparse.ArgumentParser(prog="revo", description="ONFR REVO — Restore without altering.")
    ap.add_argument("imagen")
    ap.add_argument("-m", "--modo", default="mejorar_restaurar", choices=[m for m in MODES if m != "restaurar_cuadro"])
    ap.add_argument("-i", "--intensidad", type=int, default=35, help="0 (conservadora) a 100 (intensa)")
    ap.add_argument("-x", "--escala", type=int, default=2, choices=[1, 2, 4])
    ap.add_argument("-o", "--salida")
    ap.add_argument("--mapa", help="guardar también el mapa de intervención")
    ap.add_argument("--color", type=int, default=0, help="colorear (estimación): intensidad 1-100; 0 = no")
    ap.add_argument("--mascara", help="imagen en blanco y negro con los daños pintados en blanco")
    a = ap.parse_args()

    bgr = cv2.imread(a.imagen, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"No se puede leer {a.imagen}")
    res = process(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        Settings(
            mode=a.modo, intensity=a.intensidad / 100, scale=a.escala,
            colorize=a.color > 0, color_amount=a.color / 100,
            user_mask=cv2.imread(a.mascara, cv2.IMREAD_GRAYSCALE) if a.mascara else None,
        ),
        progress=lambda f, m: print(f"[{f:4.0%}] {m}"),
    )
    out = a.salida or os.path.splitext(a.imagen)[0] + "_revo.png"
    cv2.imwrite(out, cv2.cvtColor(res.image, cv2.COLOR_RGB2BGR))
    if a.mapa:
        cv2.imwrite(a.mapa, cv2.cvtColor(intervention_map(res)[0], cv2.COLOR_RGB2BGR))
    print()
    print("\n".join(l.replace("**", "").replace("_", "").rstrip() for l in summary_lines(res)))
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
