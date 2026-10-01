# ONFR REVO — prototipo 0.1

**Restore without altering.** · _Mejora la imagen. Conserva el original._

Prototipo local en Python con interfaz web (Gradio). Incluye los modos **Mejorar**, **Restaurar** y **Mejorar + Restaurar** (por defecto), protección facial siempre activa con verificación, control de intensidad, comparador antes/después, resumen de lo que se ha tocado y mapa de intervención. **Restaurar cuadro** y la **colorización** aparecen en la interfaz como «próximamente» (fase 2) y la estructura del código ya las prevé.

## Cómo ejecutarlo

Requiere Python 3.10 o superior.

```bash
cd onfr-revo
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python setup_models.py        # descarga los modelos faciales (~130 MB, una sola vez)
python app.py                 # abre http://127.0.0.1:7860
```

Si `dlib-bin` no tiene binario para tu sistema, instala `dlib` (necesita CMake y un compilador de C++).

### Línea de comandos

```bash
python -m revo foto.jpg                                   # Mejorar + Restaurar, intervención 35
python -m revo foto.jpg -m restaurar -i 60 --mapa mapa.png
python -m revo foto.jpg -m mejorar -x 4 -o salida.png
```

### Pruebas

```bash
python tests/make_samples.py   # genera fotos degradadas de prueba en samples/
python tests/test_revo.py
```

## Cómo trabaja (y por qué no inventa)

| Paso | Qué hace | Por qué es fiel |
|---|---|---|
| Análisis | Mide ruido, contraste, blanco y negro, compresión y posibles daños | Solo se muestran las opciones que tienen sentido |
| Rostros | Detecta caras y 68 puntos faciales (dlib) | Las caras se tratan aparte desde el principio |
| Polvo y arañazos | Detecta motas y líneas finas sobre zonas lisas y las rellena con su entorno inmediato | Donde hay textura (pelo, tejidos) no toca nada. **Nunca** rellena sobre ojos, cejas, nariz o boca. En la piel solo quita motas claras diminutas, porque una mancha oscura puede ser un lunar |
| Ruido | Reducción no local de ruido; conserva parte del grano | Una foto de 1930 sigue pareciendo de 1930 |
| Luz y contraste | Ajuste global suave y un poco de contraste local | Moderado y escalado con la intensidad |
| Resolución y nitidez | Escalado Lanczos y refuerzo de bordes con control de halos | Solo refuerza bordes que ya existen; no genera textura |
| **Fidelity** | Cada cara se compara con su versión mínima: identidad (vector de reconocimiento facial), posición de los 68 puntos (rasgos y expresión) y estructura | Si no pasa, se reduce la mejora en esa cara paso a paso hasta dejarla como el original. Mejor algo borroso que inventado |

El resumen final indica qué se ha tocado: defectos reparados, marcas sobre rostros que se han dejado intactas, el grano conservado, el porcentaje de retoques locales y, para cada cara, cuánto ha cambiado frente a su límite. El **mapa de intervención** lo muestra en la propia imagen: el calor indica retoques locales, el magenta los defectos rellenados y el recuadro verde o naranja las caras protegidas.

La prueba `test_guard_rejects_altered_face` simula una «mejora» que cambia la expresión (sonrisa más ancha y ojos más grandes) y comprueba que el verificador la rechaza: la identidad cambia 0,39 cuando el límite es 0,09, y los rasgos se mueven un 3,4 % cuando el límite es un 3 %.

## Lo que hay que saber de este prototipo

- **Sin modelos generativos.** El entorno donde se construyó no tenía acceso a los repositorios de modelos (GitHub ni Hugging Face), así que la mejora y la restauración son clásicas (OpenCV). La «IA» actual es la parte facial de dlib: detección, puntos faciales e identidad. Por diseño eso encaja con la filosofía, pero un modelo de superresolución daría más detalle en fondos y objetos. Para conectarlo, `enhance.upscale` es el punto de entrada, y la verificación facial ya está pensada para frenar a un modelo así si altera una cara.
- **El detector de polvo es heurístico.** En fotos limpias puede confundir algún brillo pequeño con polvo (en las fotos limpias de prueba, entre el 0,04 % y el 0,6 % de la imagen). Por eso no se usa en el modo Mejorar, y cada punto rellenado aparece en el mapa de intervención.
- **Daños grandes** (roturas, zonas perdidas) no se reconstruyen: quedan como están.
- **En fotos con mucho grano las caras reciben menos mejora** que el fondo, porque la verificación lo frena. Es intencionado.
- **Licencia de modelos:** el modelo de 68 puntos de dlib se entrenó con el conjunto iBUG 300-W, que solo permite uso no comercial. Antes de comercializar REVO habría que sustituirlo (por ejemplo, por el modelo de 5 puntos más otro detector de rasgos con licencia libre).

## Estructura

```
app.py              interfaz web
revo/
  pipeline.py       orquestación, modos, protección facial y resumen
  analysis.py       análisis automático de la imagen
  faces.py          detección, puntos faciales, identidad y máscaras
  fidelity.py       verificación de identidad, rasgos y estructura
  restore.py        polvo, arañazos, ruido y tono
  enhance.py        escalado y nitidez sin inventar
  __main__.py       línea de comandos
setup_models.py     descarga de modelos
tests/              imágenes de prueba y pruebas de fidelidad
```

## Fase 2 (preparada, no implementada)

- **Restaurar cuadro**: el modo existe en `MODES` y está marcado en `PHASE_2`. Necesita su propio detector de grietas (craquelado) que respete la pincelada.
- **Colorización**: el ajuste `Settings.colorize` ya existe. Se aplicaría como capa separada y etiquetada como estimación, nunca mezclada con la restauración.
