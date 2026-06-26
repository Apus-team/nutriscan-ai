# Informe Técnico - NutriScan AI

## Problema

Identificar automáticamente alimentos en imágenes y proporcionar información calórica estimada, utilizando técnicas de visión computacional y aprendizaje profundo.

## Objetivo

Desarrollar un clasificador de imágenes de alimentos en versión Beta, capaz de distinguir entre 10 categorías del dataset Food-101, con una interfaz web interactiva que permita carga de imágenes y captura por cámara.

## Dataset

Se utiliza un subconjunto del dataset Food-101, limitado a las siguientes 10 clases:

- apple_pie, hamburger, french_fries, hot_dog, sushi, donuts, ice_cream, chicken_wings, pizza, caesar_salad

Cada clase contiene aproximadamente 750 imágenes de entrenamiento y 250 de prueba.

## Arquitectura del modelo

| Componente | Descripción |
|------------|-------------|
| Modelo base | EfficientNetB0 (pre-entrenado en ImageNet) |
| Input shape | 224 × 224 × 3 |
| Data Augmentation | Capas internas del modelo (RandomFlip, Rotation, Zoom, Translation, Contraste) |
| Normalización | EfficientNetB0 preprocess_input dentro del grafo |
| Capas adicionales | GlobalAveragePooling2D → Dropout(0.5) → Dense(10, softmax) |
| Congelamiento (Fase 1) | Backbone congelado; solo se entrena la cabeza clasificadora |
| Fine-tuning (Fase 2) | Backbone descongelado con learning rate 1e-5 |

## Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Optimizador | Adam (Fase 1: lr=0.001, Fase 2: lr=1e-5) |
| Función de pérdida | CategoricalCrossentropy |
| Batch size | 32 |
| Épocas | 35 (15 Feature Extraction + 20 Fine-Tuning) |
| Callbacks | EarlyStopping (paciencia 5/7), ReduceLROnPlateau, ModelCheckpoint |
| Data augmentation | Integrada en el modelo: Flip, Rotation 20%, Zoom 20%, Translation 10%, Contraste 15% |
| Preprocesamiento | Resize a 224×224; normalización interna via EfficientNetB0 preprocess_input |

## Métricas

> *Las siguientes métricas deben completarse después del entrenamiento del modelo.*

| Métrica | Valor |
|---------|-------|
| Accuracy (test) | — |
| Precision (macro) | — |
| Recall (macro) | — |
| F1-Score (macro) | — |

### Matriz de confusión

> *Insertar imagen de la matriz de confusión después del entrenamiento.*

```
[Imagen: report/images/confusion_matrix.png]
```

## Interfaz de usuario

La aplicación Streamlit ofrece:

- Título y descripción del proyecto
- Selección entre carga de archivo o captura por cámara
- Previsualización de la imagen seleccionada
- Predicción con nombre del alimento, confianza porcentual y calorías aproximadas
- Advertencia visual si la confianza es menor al 60%
- Manejo de errores para imágenes inválidas y modelo ausente

## Control de errores

| Escenario | Respuesta |
|-----------|-----------|
| Imagen vacía | Mensaje informativo para subir/tomar una foto |
| Formato inválido | Error controlado con mensaje al usuario |
| Modelo ausente | Advertencia temprana con indicación de entrenar el modelo |
| Confianza < 60% | Advertencia visual de baja confianza |
| Clase no reconocida | El sistema solo predice dentro de las 10 clases entrenadas |

## Limitaciones

1. **Catálogo cerrado**: solo reconoce las 10 clases entrenadas.
2. **Calorías estimadas**: valores aproximados por porción estándar; no sustituyen consejo nutricional profesional.
3. **Sensibilidad visual**: la calidad de la predicción depende de iluminación, ángulo, fondo y resolución.
4. **Beta funcional**: no debe usarse en entornos críticos sin validación humana.
5. **Tamaño del modelo**: EfficientNetB0 genera un archivo ~20 MB, manejable para despliegue en Streamlit Cloud.

## Conclusiones

NutriScan AI demuestra la viabilidad de un clasificador de alimentos basado en transfer learning con EfficientNetB0. La arquitectura permite predicciones rápidas y precisas dentro del dominio acotado de 10 clases. La integración de data augmentation como capas del modelo permite ejecución en GPU y se desactiva automáticamente en inferencia. La interfaz Streamlit facilita la interacción del usuario final mediante carga de imágenes y captura por cámara.

El proyecto está estructurado para escalar: añadir nuevas clases requiere actualizar el dataset, reentrenar y modificar `labels.json` y `calories.json`.

### Trabajo futuro

- Ampliar el catálogo a las 101 clases completas de Food-101.
- Implementar modelo de segmentación para aislar el alimento del fondo.
- Integrar base de datos nutricional real (USDA).
- Añadir modo lote para múltiples imágenes.
- Desplegar con Docker para entornos de producción.
