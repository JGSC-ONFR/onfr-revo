"""Descarga los modelos faciales de dlib (≈130 MB) desde PyPI
(paquete face_recognition_models) y los deja en ./models.

    python setup_models.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "models")
NEEDED = [
    "shape_predictor_68_face_landmarks.dat",
    "shape_predictor_5_face_landmarks.dat",
    "dlib_face_recognition_resnet_model_v1.dat",
    "mmod_human_face_detector.dat",
]


def main():
    os.makedirs(DEST, exist_ok=True)
    if all(os.path.exists(os.path.join(DEST, n)) for n in NEEDED):
        print("Los modelos ya están en", DEST)
        return
    with tempfile.TemporaryDirectory() as tmp:
        print("Descargando face_recognition_models desde PyPI…")
        subprocess.check_call([
            sys.executable, "-m", "pip", "download", "face_recognition_models==0.3.0",
            "--no-deps", "--no-binary", ":all:", "-d", tmp,
        ])
        archive = glob.glob(os.path.join(tmp, "face_recognition_models-*.tar.gz"))[0]
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                name = os.path.basename(m.name)
                if name in NEEDED:
                    with tar.extractfile(m) as src, open(os.path.join(DEST, name), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print("  ✓", name)
    missing = [n for n in NEEDED if not os.path.exists(os.path.join(DEST, n))]
    if missing:
        sys.exit(f"Faltan: {missing}")
    print("Listo.")


if __name__ == "__main__":
    main()
