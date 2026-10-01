"""Análisis automático de la imagen: decide qué opciones son relevantes."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Analysis:
    width: int
    height: int
    monochrome: bool  # blanco y negro o virado (sepia)
    noise: float  # sigma estimado del ruido (0-255)
    sharpness: float  # varianza del laplaciano normalizada
    defects: float  # fracción aproximada de píxeles con polvo/arañazos
    contrast: float  # rango dinámico p1-p99 (0-255)
    jpeg_blockiness: float

    def suggested_mode(self) -> str:
        if self.defects > 0.003 or self.monochrome:
            return "mejorar_restaurar"
        return "mejorar"

    def describe(self) -> list[str]:
        out = [f"Resolución: {self.width} × {self.height}"]
        out.append("Imagen en blanco y negro / virada" if self.monochrome else "Imagen en color")
        if self.defects > 0.003:
            out.append("Se detectan polvo, manchas o arañazos")
        if self.noise > 6:
            out.append("Ruido o grano apreciable")
        if self.contrast < 170:
            out.append("Contraste reducido")
        if self.jpeg_blockiness > 1.25:
            out.append("Artefactos de compresión")
        if self.sharpness < 60:
            out.append("Poco detalle / desenfoque")
        return out


def estimate_noise(gray: np.ndarray) -> float:
    """Estimación robusta del sigma del ruido (Immerkær, 1996)."""
    g = gray.astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    conv = cv2.filter2D(g, -1, k)
    # descarta bordes fuertes para no confundir textura con ruido
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    vals = np.abs(conv[edges == 0])
    if vals.size == 0:
        vals = np.abs(conv)
    return float(np.sqrt(np.pi / 2) * vals.mean() / 6.0)


def blockiness(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))
    if dx.shape[1] < 16:
        return 1.0
    at8 = dx[:, 7::8].mean()
    other = dx.mean()
    return float(at8 / (other + 1e-6))


def is_monochrome(rgb: np.ndarray) -> bool:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    # una imagen virada tiene croma casi constante; una en color, variada
    return float(np.std(a) + np.std(b)) < 6.0


def analyze(rgb: np.ndarray) -> Analysis:
    from .restore import detect_defects

    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    noise = estimate_noise(gray)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    p1, p99 = np.percentile(gray, [1, 99])
    defects = detect_defects(rgb, noise)
    return Analysis(
        width=w,
        height=h,
        monochrome=is_monochrome(rgb),
        noise=noise,
        sharpness=float(lap / max(1.0, noise)),
        defects=float((defects > 0).mean()),
        contrast=float(p99 - p1),
        jpeg_blockiness=blockiness(gray),
    )
