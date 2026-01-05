# 🚀 Guía de Inicio Rápido - quickstart.py

Esta guía te ayudará a ejecutar el script `quickstart.py` paso a paso, desde la configuración inicial hasta la obtención de resultados.

## 📋 Tabla de Contenidos

1. [¿Qué es quickstart.py?](#qué-es-quickstartpy)
2. [Requisitos Previos](#requisitos-previos)
3. [Configuración Inicial](#configuración-inicial)
4. [Modo CSV (Sin API)](#modo-csv-sin-api)
5. [Modo API (Con API Key)](#modo-api-con-api-key)
6. [Ejecución Paso a Paso](#ejecución-paso-a-paso)
7. [Entendiendo los Resultados](#entendiendo-los-resultados)
8. [Solución de Problemas](#solución-de-problemas)

---

## ¿Qué es quickstart.py?

`quickstart.py` es un script de inicio rápido que automatiza todo el proceso de:

1. **Extracción de Datos**: Obtiene datos de partidos de la Premier League
2. **Procesamiento**: Limpia y transforma los datos
3. **Feature Engineering**: Crea 38 características predictivas
4. **Entrenamiento del Modelo**: Entrena un modelo Poisson para predecir Over/Under 2.5 goles
5. **Validación Cruzada**: Evalúa el modelo con 5-fold cross-validation
6. **Guardado**: Guarda el modelo entrenado y genera visualizaciones

**Tiempo estimado**: 2-5 minutos (dependiendo del modo)

---

## Requisitos Previos

### 1. Python y Dependencias

Asegúrate de tener Python 3.8+ instalado y todas las dependencias:

```bash
# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete en modo desarrollo
pip install -e .
```

### 2. Estructura de Directorios

El script creará automáticamente estos directorios si no existen:
- `data/raw/` - Datos sin procesar
- `data/processed/` - Datos transformados
- `data/final/` - Datos finales para entrenamiento
- `models/` - Modelos entrenados
- `models/plots/` - Gráficos de evaluación
- `models/results/` - Resultados de validación cruzada
- `logs/` - Archivos de log

---

## Configuración Inicial

### Opción 1: Modo CSV (Recomendado para empezar)

**No requiere configuración adicional.** El script descargará automáticamente datos históricos de CSV.

### Opción 2: Modo API (Requiere API Key)

Si quieres usar datos en tiempo real de la API, necesitas:

1. **Obtener una API Key de RapidAPI**
   - Visita: https://rapidapi.com/api-sports/api/api-football
   - Regístrate (hay un plan gratuito con 100 requests/día)
   - Copia tu API key

2. **Crear archivo `.env`**

   En la raíz del proyecto, crea un archivo `.env` con:

   ```env
   # API Configuration
   FOOTBALL_DATA_API_KEY=tu_api_key_aqui
   
   # League Settings (opcional, valores por defecto)
   LEAGUE_ID=39
   CURRENT_SEASON=2024-2025
   ```

   **Ejemplo real:**
   ```env
   FOOTBALL_DATA_API_KEY=abc123def456ghi789jkl012mno345pqr678
   LEAGUE_ID=39
   CURRENT_SEASON=2024-2025
   ```

3. **Verificar que el archivo existe**

   ```bash
   # Windows (PowerShell)
   Test-Path .env
   
   # Linux/Mac
   ls -la .env
   ```

---

## Modo CSV (Sin API)

### ¿Cuándo usar este modo?

- ✅ Es tu primera vez ejecutando el script
- ✅ No tienes API key
- ✅ Quieres datos históricos (últimas 3 temporadas)
- ✅ No necesitas datos en tiempo real

### Ejecución

```bash
python quickstart.py
```

### ¿Qué hace?

1. Descarga datos de 3 temporadas: 2022-2023, 2023-2024, 2024-2025
2. Procesa y limpia los datos
3. Crea características predictivas
4. Entrena el modelo
5. Ejecuta validación cruzada
6. Guarda el modelo y genera gráficos

### Ventajas

- ✅ No requiere API key
- ✅ Datos históricos completos
- ✅ Más rápido (sin límites de rate limiting)
- ✅ Funciona offline después de la primera descarga

---

## Modo API (Con API Key)

### ¿Cuándo usar este modo?

- ✅ Tienes una API key válida
- ✅ Necesitas datos de la temporada actual (2024)
- ✅ Quieres datos actualizados en tiempo real

### Configuración

1. **Crea el archivo `.env`** (ver sección anterior)

2. **Verifica tu API key**

   ```bash
   # Windows (PowerShell)
   $env:FOOTBALL_DATA_API_KEY
   
   # Linux/Mac
   echo $FOOTBALL_DATA_API_KEY
   ```

### Ejecución

```bash
python quickstart.py --api
```

### ¿Qué hace?

1. Conecta a la API de RapidAPI
2. Descarga datos de la temporada 2024
3. Procesa y limpia los datos
4. Crea características predictivas
5. Entrena el modelo
6. Ejecuta validación cruzada
7. Guarda el modelo y genera gráficos

### Limitaciones

- ⚠️ Plan gratuito: 100 requests/día
- ⚠️ Rate limiting: 10 requests/minuto (automático)
- ⚠️ Solo datos de la temporada actual

---

## Ejecución Paso a Paso

### Paso 1: Preparación

```bash
# Navega al directorio del proyecto
cd premier-league-predictor

# Activa tu entorno virtual (si usas uno)
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Paso 2: Verificar Instalación

```bash
# Verifica que Python puede importar los módulos
python -c "from pipelines import DataPipeline, ModelPipeline; print('OK')"
```

Si ves "OK", estás listo. Si hay errores, instala las dependencias:

```bash
pip install -r requirements.txt
pip install -e .
```

### Paso 3: Ejecutar el Script

**Modo CSV (recomendado para empezar):**
```bash
python quickstart.py
```

**Modo API:**
```bash
python quickstart.py --api
```

### Paso 4: Observar el Progreso

Verás un output similar a:

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ╔═╗╦═╗╔═╗╔╦╗╦ ╦╔═╗╦═╗╔═╗╔═╗╦═╗╔═╗╔═╗╦═╗╔═╗╔╦╗╔═╗╦═╗    ║
║     ╠═╝╠╦╝║ ║ ║ ╠═╣║ ║╠╦╝║╣ ║ ╦╠╦╝║╣ ║ ║╠╦╝╠═╣ ║ ║ ║╠╦╝    ║
║     ╩  ╩╚═╚═╝ ╩ ╩ ╩╚═╝╩╚═╚═╝╚═╝╩╚═╚═╝╚═╝╩╚═╩ ╩ ╩ ╚═╝╩╚═    ║
║                                                               ║
║              🚀 Quick Start Pipeline 🚀              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

======================================================================
⚙️ Setting Up Logging
======================================================================

✅ Logging configured successfully

======================================================================
⚙️ Initializing Pipelines
======================================================================

✅ Pipelines imported successfully

======================================================================
📁 PART 1: Data Pipeline
======================================================================

⏳ [1/6] Initializing DataPipeline...
✅ DataPipeline initialized
⏳ [2/6] Running data extraction and processing...
ℹ️ Mode: CSV (3 seasons: 2223, 2324, 2425)
...
```

### Paso 5: Esperar Completación

El script mostrará:
- ✅ Progreso de cada etapa
- 📊 Métricas finales
- 📁 Archivos generados
- ⭐ Próximos pasos sugeridos

---

## Entendiendo los Resultados

### Métricas Principales

El script muestra estas métricas:

1. **ROC-AUC** (0.0 - 1.0)
   - Mide la capacidad del modelo de distinguir entre clases
   - > 0.65 es bueno para predicciones deportivas
   - > 0.70 es excelente

2. **Brier Score** (0.0 - 1.0)
   - Mide la calibración de probabilidades
   - Más bajo es mejor
   - < 0.25 es bueno

3. **Log Loss** (0.0 - ∞)
   - Mide la calidad de las probabilidades
   - Más bajo es mejor
   - < 0.7 es bueno

4. **Accuracy** (0.0 - 1.0)
   - Porcentaje de predicciones correctas
   - > 0.60 es bueno para Over/Under

5. **Precision, Recall, F1 Score**
   - Miden el rendimiento por clase
   - Útiles para entender errores del modelo

### Resultados de Validación Cruzada

Muestra la media y desviación estándar de 5 folds:
- **Mean ± Std**: Indica la consistencia del modelo
- **Std bajo**: Modelo más estable
- **Std alto**: Puede indicar overfitting o datos inconsistentes

### Archivos Generados

```
models/
├── poisson_model_latest.pkl          # Modelo entrenado
├── poisson_model_latest_metadata.json # Metadatos del modelo
├── poisson_model_YYYYMMDD_HHMMSS.pkl  # Versión timestamped
├── plots/
│   ├── calibration_curve.png          # Calibración de probabilidades
│   ├── roc_curve.png                  # Curva ROC
│   └── confusion_matrix.png           # Matriz de confusión
└── results/
    └── cv_results.csv                  # Resultados de CV

data/final/
└── training_data_latest.parquet       # Datos de entrenamiento
```

---

## Solución de Problemas

### Error: "No module named 'src'"

**Solución:**
```bash
# Asegúrate de estar en el directorio raíz
cd premier-league-predictor

# Instala el paquete
pip install -e .
```

### Error: "API key not found"

**Solución:**
1. Verifica que el archivo `.env` existe en la raíz del proyecto
2. Verifica que contiene `FOOTBALL_DATA_API_KEY=tu_key`
3. O usa el modo CSV: `python quickstart.py` (sin `--api`)

### Error: "Rate limit exceeded"

**Solución:**
- El script espera automáticamente, pero si persiste:
- Espera unos minutos y vuelve a intentar
- O usa el modo CSV que no tiene límites

### Error: "training_data_latest.parquet not found"

**Solución:**
- Esto no debería pasar, el script crea los datos automáticamente
- Si ocurre, ejecuta solo el data pipeline primero:
  ```python
  from pipelines import DataPipeline
  pipeline = DataPipeline()
  pipeline.run_full_pipeline(source='csv')
  ```

### El script es muy lento

**Causas posibles:**
- Primera ejecución (descarga datos)
- Modo API con rate limiting
- Muchos datos (3 temporadas)

**Soluciones:**
- Usa modo CSV (más rápido)
- Espera la primera ejecución (datos se guardan)
- Revisa los logs en `logs/` para ver qué está tardando

### Colores no funcionan en Windows

**Solución:**
- Es normal en algunos terminales de Windows
- Usa `--no-colors` para desactivar colores:
  ```bash
  python quickstart.py --no-colors
  ```

---

## Próximos Pasos

Después de ejecutar `quickstart.py` exitosamente:

1. **Revisa los gráficos** en `models/plots/`
   - `calibration_curve.png`: ¿Están bien calibradas las probabilidades?
   - `roc_curve.png`: ¿Qué tan bien distingue el modelo?
   - `confusion_matrix.png`: ¿Qué tipos de errores comete?

2. **Explora el modelo**
   ```python
   from src.models import PoissonGoalsModel
   
   model = PoissonGoalsModel.load('models/poisson_model_latest.pkl')
   print(model.get_model_summary())
   ```

3. **Haz predicciones**
   ```python
   # Carga datos nuevos
   import pandas as pd
   new_data = pd.read_parquet('data/final/training_data_latest.parquet')
   
   # Predice
   predictions = model.predict(new_data.head(10))
   probabilities = model.predict_proba(new_data.head(10))
   ```

4. **Experimenta**
   - Modifica features en `src/features/engineering.py`
   - Prueba diferentes thresholds
   - Ajusta hiperparámetros

5. **Lee la documentación**
   - `QUICKSTART.md` - Guía general del proyecto
   - `README.md` - Documentación principal
   - `src/*/README.md` - Documentación de módulos

---

## Resumen de Comandos

```bash
# Modo CSV (sin API key)
python quickstart.py

# Modo API (requiere API key en .env)
python quickstart.py --api

# Sin colores (para terminales que no los soportan)
python quickstart.py --no-colors

# Ver ayuda
python quickstart.py --help
```

---

## Preguntas Frecuentes

**P: ¿Necesito una API key para empezar?**
R: No. El modo CSV funciona sin API key.

**P: ¿Cuánto tiempo tarda?**
R: 2-5 minutos dependiendo del modo y tu conexión.

**P: ¿Puedo usar mis propios datos?**
R: Sí, pero necesitarías modificar el código. Mejor usa los pipelines directamente.

**P: ¿El modelo es bueno?**
R: Para Over/Under 2.5, 60-65% accuracy es bueno. El fútbol es impredecible.

**P: ¿Puedo usar esto en producción?**
R: El script es para desarrollo. Para producción, usa los pipelines directamente con validación adicional.

---

¡Feliz predicción! ⚽📈

