# NutriScan AI

Clasificador de alimentos basado en visión computacional con TensorFlow/Keras y Streamlit.

## Descripción

NutriScan AI es un sistema Beta de clasificación de imágenes de alimentos. Reconoce 10 clases del dataset Food-101 y muestra las calorías aproximadas del alimento detectado.

## Instalación

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## Ejecución local

```bash
streamlit run app.py
```

## Estructura del proyecto

```
nutriscan-ai/
├── app.py                  # Aplicación Streamlit
├── requirements.txt        # Dependencias
├── README.md               # Documentación
├── calories.json           # Calorías por alimento
├── labels.json             # Mapeo índice → clase
├── .gitignore
├── model/
│   └── nutriscan_model.keras  # Modelo entrenado (generado)
├── notebooks/
│   └── entrenamiento_nutriscan.ipynb
├── src/
│   ├── preprocess.py       # Preprocesamiento de imágenes
│   ├── predict.py          # Carga y predicción del modelo
│   └── utils.py            # Utilidades (labels, calorías)
└── report/
    └── informe_nutriscan.md
```

## Arquitectura del modelo

- Modelo base: EfficientNetB0 (transfer learning)
- Data augmentation: capas internas del modelo (Flip, Rotation, Zoom, Translation, Contraste)
- Normalización: EfficientNetB0 preprocess_input dentro del grafo
- Capas: GlobalAveragePooling2D + Dropout(0.5) + Dense(10, softmax)
- Entrada: 224×224×3
- Fase 1 (Feature Extraction): backbone congelado, Adam lr=0.001
- Fase 2 (Fine-Tuning): backbone descongelado, Adam lr=1e-5
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Alimentos reconocidos

| Clase | Calorías (aprox.) |
|-------|-------------------|
| Apple Pie | 400 kcal |
| Hamburger | 550 kcal |
| French Fries | 365 kcal |
| Hot Dog | 290 kcal |
| Sushi | 200 kcal |
| Donuts | 300 kcal |
| Ice Cream | 270 kcal |
| Chicken Wings | 430 kcal |
| Pizza | 400 kcal |
| Caesar Salad | 180 kcal |

## Limitaciones

- Reconoce exclusivamente las 10 clases del catálogo entrenado.
- La confianza puede verse afectada por iluminación, ángulo y fondo.
- En etapa Beta: las predicciones deben validarse manualmente.
- Las calorías son estimaciones por porción estándar, no personalizadas.

## Despliegue en Streamlit Cloud

1. Sube el repositorio a GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io).
3. Conecta tu repositorio.
4. Configura `app.py` como archivo de entrada.
5. Agrega `model/nutriscan_model.keras` usando [Git LFS](https://git-lfs.com) o genera el modelo localmente y súbelo manualmente.

## Licencia

MIT
