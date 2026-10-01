"""Detección y descripción de rostros (FaceGuard).

Usa los modelos de dlib (no generativos):
  - detector HOG integrado (y CNN MMOD como segundo intento),
  - 68 puntos faciales (geometría y expresión),
  - ResNet de reconocimiento facial (vector de identidad de 128 dimensiones).

Nada de este módulo genera píxeles: solo localiza y mide caras para que el
resto del pipeline pueda protegerlas y verificar que no han cambiado.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import dlib
import numpy as np

MODELS_DIR = os.environ.get(
    "REVO_MODELS", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
)

# Índices de los 68 puntos (convención iBUG 300-W)
JAW = list(range(0, 17))
BROWS = list(range(17, 27))
NOSE = list(range(27, 36))
EYE_L = list(range(36, 42))
EYE_R = list(range(42, 48))
MOUTH = list(range(48, 68))
FEATURES = BROWS + NOSE + EYE_L + EYE_R + MOUTH


@dataclass
class Face:
    rect: tuple[int, int, int, int]  # x0, y0, x1, y1 en coordenadas de la imagen original
    landmarks: np.ndarray  # (68, 2) float
    embedding: np.ndarray | None = None  # (128,)
    extra: dict = field(default_factory=dict)

    @property
    def interocular(self) -> float:
        l = self.landmarks[EYE_L].mean(0)
        r = self.landmarks[EYE_R].mean(0)
        return float(np.linalg.norm(l - r)) + 1e-6

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]


class FaceGuard:
    _instance = None

    def __init__(self, models_dir: str = MODELS_DIR):
        def p(name):
            path = os.path.join(models_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Falta el modelo {name} en {models_dir}. Ejecuta: python setup_models.py"
                )
            return path

        self.hog = dlib.get_frontal_face_detector()
        self.cnn_path = os.path.join(models_dir, "mmod_human_face_detector.dat")
        self._cnn = None
        self.shape68 = dlib.shape_predictor(p("shape_predictor_68_face_landmarks.dat"))
        self.shape5 = dlib.shape_predictor(p("shape_predictor_5_face_landmarks.dat"))
        self.recog = dlib.face_recognition_model_v1(p("dlib_face_recognition_resnet_model_v1.dat"))

    @classmethod
    def get(cls) -> "FaceGuard":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ detección
    def detect(self, rgb: np.ndarray) -> list[Face]:
        """Detecta caras en una imagen RGB uint8. Trabaja sobre una copia
        reducida y normalizada en contraste (solo para detectar; no modifica
        la imagen). Primero a 800 px (rápido); si no encuentra nada, repite
        con más resolución y, si hace falta, con el detector CNN."""
        h, w = rgb.shape[:2]
        rects, scale = [], 1.0
        for side in (800, 1600):
            scale = min(1.0, side / max(h, w))
            small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
            gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
            rects = list(self.hog(gray, 1))
            if rects or scale == 1.0:
                break
        if not rects and os.path.exists(self.cnn_path):
            scale = min(1.0, 900.0 / max(h, w))
            small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
            gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
            if self._cnn is None:
                self._cnn = dlib.cnn_face_detection_model_v1(self.cnn_path)
            rects = [d.rect for d in self._cnn(gray, 1) if d.confidence > 0.5]

        faces = []
        for r in rects:
            x0, y0 = int(max(0, r.left() / scale)), int(max(0, r.top() / scale))
            x1, y1 = int(min(w, r.right() / scale)), int(min(h, r.bottom() / scale))
            if x1 - x0 < 12 or y1 - y0 < 12:
                continue
            face = Face(rect=(x0, y0, x1, y1), landmarks=np.zeros((68, 2)))
            self.describe(rgb, face)
            faces.append(face)
        return faces

    # ------------------------------------------------------------- descripción
    def landmarks(self, rgb: np.ndarray, rect) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        r = dlib.rectangle(*[int(v) for v in rect])
        s = self.shape68(gray, r)
        return np.array([[p.x, p.y] for p in s.parts()], dtype=np.float64)

    def embedding(self, rgb: np.ndarray, rect) -> np.ndarray | None:
        try:
            r = dlib.rectangle(*[int(v) for v in rect])
            s = self.shape5(rgb, r)
            chip = dlib.get_face_chip(rgb, s, size=150, padding=0.25)
            return np.array(self.recog.compute_face_descriptor(chip))
        except Exception:
            return None

    def describe(self, rgb: np.ndarray, face: Face) -> Face:
        face.landmarks = self.landmarks(rgb, face.rect)
        face.embedding = self.embedding(rgb, face.rect)
        return face


# ---------------------------------------------------------------- máscaras
def face_masks(shape, face: Face, scale: float = 1.0):
    """Devuelve (mascara_cara, mascara_rasgos) float32 en [0,1] con borde suave.

    - mascara_cara: óvalo que cubre la cara completa (frente incluida).
    - mascara_rasgos: ojos, cejas, nariz y boca, dilatados. Zona de máxima
      protección: aquí nunca se rellena ni se reconstruye nada.
    """
    h, w = shape[:2]
    pts = face.landmarks * scale
    io = face.interocular * scale

    hull_pts = np.concatenate([pts[JAW], pts[BROWS] - [0, 0.6 * io]])
    hull = cv2.convexHull(hull_pts.astype(np.int32))
    face_m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(face_m, hull, 255)
    k = max(3, int(io * 0.25)) | 1
    face_m = cv2.dilate(face_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    feat_m = np.zeros((h, w), np.uint8)
    for group in (BROWS[:5], BROWS[5:], NOSE, EYE_L, EYE_R, MOUTH):
        cv2.fillConvexPoly(feat_m, cv2.convexHull(pts[group].astype(np.int32)), 255)
    k2 = max(3, int(io * 0.18)) | 1
    feat_m = cv2.dilate(feat_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2)))

    blur = max(3, int(io * 0.3)) | 1
    face_f = cv2.GaussianBlur(face_m.astype(np.float32) / 255.0, (blur, blur), 0)
    feat_f = cv2.GaussianBlur(feat_m.astype(np.float32) / 255.0, (blur, blur), 0)
    return face_f, np.maximum(feat_f, (feat_m > 0).astype(np.float32))
