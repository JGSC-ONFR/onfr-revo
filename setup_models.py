"""Descarga los modelos de REVO en ./models (una sola vez).

    python setup_models.py

- Rostros (obligatorio, ~130 MB): modelos de dlib, desde PyPI
  (paquete face_recognition_models).
- Color (opcional, ~130 MB): Zhang et al. 2016, «Colorful Image Colorization».
- Relleno IA (opcional, ~200 MB): LaMa en formato ONNX.

Si un modelo opcional no se puede descargar, REVO funciona igual: sin
colorización, o con el relleno clásico en lugar de LaMa.
"""
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "models")
FACES = [
    "shape_predictor_68_face_landmarks.dat",
    "shape_predictor_5_face_landmarks.dat",
    "dlib_face_recognition_resnet_model_v1.dat",
    "mmod_human_face_detector.dat",
]
HF_COLOR = "https://huggingface.co/spaces/BilalSardar/Black-N-White-To-Color/resolve/main/"
GH_COLOR = "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/"
OPTIONAL = {
    "colorization_deploy_v2.prototxt": [HF_COLOR + "colorization_deploy_v2.prototxt", GH_COLOR + "models/colorization_deploy_v2.prototxt"],
    "pts_in_hull.npy": [HF_COLOR + "pts_in_hull.npy", GH_COLOR + "resources/pts_in_hull.npy"],
    "colorization_release_v2.caffemodel": [
        HF_COLOR + "colorization_release_v2.caffemodel",
        "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1",
    ],
    "lama_fp32.onnx": ["https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"],
}


def faces():
    if all(os.path.exists(os.path.join(DEST, n)) for n in FACES):
        print("✓ Modelos faciales")
        return
    with tempfile.TemporaryDirectory() as tmp:
        print("Descargando modelos faciales desde PyPI…")
        subprocess.check_call([
            sys.executable, "-m", "pip", "download", "face_recognition_models==0.3.0",
            "--no-deps", "--no-binary", ":all:", "-d", tmp,
        ])
        archive = glob.glob(os.path.join(tmp, "face_recognition_models-*.tar.gz"))[0]
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                name = os.path.basename(m.name)
                if name in FACES:
                    with tar.extractfile(m) as src, open(os.path.join(DEST, name), "wb") as dst:
                        shutil.copyfileobj(src, dst)
    missing = [n for n in FACES if not os.path.exists(os.path.join(DEST, n))]
    if missing:
        sys.exit(f"Faltan: {missing}")
    print("✓ Modelos faciales")


def optional():
    for name, urls in OPTIONAL.items():
        path = os.path.join(DEST, name)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print("✓", name)
            continue
        for url in urls:
            try:
                print(f"Descargando {name}…")
                req = urllib.request.Request(url, headers={"User-Agent": "onfr-revo"})
                with urllib.request.urlopen(req, timeout=60) as r, open(path + ".part", "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                if os.path.getsize(path + ".part") < 1000:
                    raise OSError("archivo vacío")
                os.replace(path + ".part", path)
                print("✓", name)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  no se pudo desde {url.split('/')[2]}: {e}")
                if os.path.exists(path + ".part"):
                    os.remove(path + ".part")
        else:
            print(f"  (opcional) {name} no disponible; REVO funcionará sin él")


if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    faces()
    optional()
    print("Listo.")
