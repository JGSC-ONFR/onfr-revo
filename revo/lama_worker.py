"""Proceso aparte para la IA de relleno (LaMa).

Crear la sesión de onnxruntime bloquea el intérprete de Python (13-33 s en un
portátil): dentro del proceso de la app congelaba la interfaz y retrasaba la
apertura de REVO. Aquí se carga sin molestar y luego atiende peticiones
(recorte, hueco) → recorte rellenado, en formato pickle por stdin/stdout.
"""
from __future__ import annotations

import pickle
import sys

import cv2
import numpy as np


def run(sess, crop: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """LaMa trabaja a 512x512: imagen en [0,1] (1,3,512,512) y máscara
    (1,1,512,512) con 1 = hueco."""
    h, w = crop.shape[:2]
    img = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    m = (cv2.resize(hole.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    feed = {}
    for inp in sess.get_inputs():
        if "mask" in inp.name.lower() or (inp.shape and inp.shape[1] == 1):
            feed[inp.name] = m[None, None]
        else:
            feed[inp.name] = img.transpose(2, 0, 1)[None]
    out = sess.run(None, feed)[0][0].transpose(1, 2, 0)
    if out.max() <= 1.5:
        out = out * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


def main(model_path: str) -> None:
    import onnxruntime as ort

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # nada más que resultados por stdout
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    pickle.dump("ready", stdout)
    stdout.flush()
    while True:
        try:
            msg = pickle.load(stdin)
        except EOFError:
            break
        if msg is None:
            break
        try:
            res = run(sess, *msg)
        except Exception as e:  # noqa: BLE001
            res = e
        pickle.dump(res, stdout)
        stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1])
