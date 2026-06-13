# Análisis del Mercado Laboral IT en Panamá 🇵🇦

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini%20API-v2.5-4285F4?style=for-the-badge&logo=google-gemini&logoColor=white)](https://aistudio.google.com/)

---

### 🏫 Universidad Tecnológica de Panamá (UTP)
* **Facultad:** Facultad de Ingeniería de Sistemas Computacionales (FISC)
* **Carrera:** Licenciatura en Desarrollo y Gestión de Software
* **Curso:** Gestión de la Información (Semestre I, 2026)
* **Asignación:** Segundo Parcial — Pipeline + Visualización

### 👥 Integrantes (Grupo 4):
* **Bryan Law** — 8-1011-2459
* **Evaristo Alvarez** — 8-1005-1928
* **Fernando Jimenez** — 20-24-7669
* **Manuel Campos** — 8-1022-1118
* **Diego Gordon** — 8-1017-349

---

## 📌 1. Introducción del Proyecto

Este proyecto consiste en el desarrollo de un **Pipeline de Datos Inteligente**, un **Modelo de Aprendizaje Automático (Machine Learning)** y un **Dashboard Interactivo** diseñado para analizar el ecosistema de empleo en tecnologías de la información (IT) en la República de Panamá. 

El sistema realiza web scraping automatizado de portales de empleo, estructura los datos usando un modelo de lenguaje masivo para extraer habilidades y salarios, aplica agrupamiento no supervisado (**K-Means**) para categorizar perfiles profesionales, modela el crecimiento de la demanda tecnológica con **Regresión Lineal** y expone todo a través de una interfaz interactiva de **Streamlit** adaptable a modo oscuro y claro.

---

## 🎯 2. Justificación del Problema en Panamá

El sector tecnológico en Panamá se encuentra en constante crecimiento debido a su posición como hub logístico y financiero. Sin embargo, existe una notable brecha entre las habilidades requeridas por las empresas y los contenidos académicos universitarios. 

Este proyecto provee a la UTP una herramienta basada en ciencia de datos que monitoriza las demandas del mercado directamente de portales activos como *Konzerta* y *Computrabajo*. Esto ayuda a identificar habilidades emergentes en tiempo real antes de que queden obsoletas, optimizando el perfil de egreso de los estudiantes de la FISC.

---

## 📁 3. Estructura del Repositorio

El proyecto está organizado siguiendo las mejores prácticas de ingeniería de software para Ciencia de Datos:

```text
parcial_n2_grupo4/
├── data/
│   ├── raw/                        # Documentos crudos extraídos de scraping
│   └── processed/                  # Base de datos SQLite y CSV limpios (Pre-poblados)
├── src/
│   ├── __init__.py
│   ├── pipeline.py                 # Código de Ingesta, Scraping e Integración con Gemini 2.5
│   ├── modelo.py                   # Modelado K-Means (Clustering) y Regresión Temporal
│   └── app.py                      # Dashboard de Streamlit con visualizaciones Plotly
├── requirements.txt                # Dependencias necesarias
└── README.md                       # Documentación del proyecto (este archivo)
```

---

## 🚀 4. Guía de Instalación y Ejecución Rápida

*(Nota: Dado que la base de datos `laboral_it.db` y el CSV `vacantes_limpias.csv` ya están pre-poblados y guardados en el repositorio, **no es necesario correr el pipeline de scraping la primera vez**; puedes lanzar el Dashboard de inmediato).*

### Paso 1: Clonar el Repositorio e Ingresar
Abre la terminal de tu computadora y ejecuta:
```bash
git clone https://github.com/tu-usuario/analisis-mercado-laboral-it-panama.git
cd analisis-mercado-laboral-it-panama
```

### Paso 2: Crear el Entorno Virtual e Instalar Dependencias
* **En Windows (Símbolo del Sistema / CMD):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  ```
* **En macOS / Linux (Terminal):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

### Paso 3: Lanzar el Dashboard de Streamlit
Ejecuta el servidor web local:
```bash
streamlit run src/app.py
```
Una ventana de tu navegador predeterminada se abrirá automáticamente en la dirección: `http://localhost:8501`.

---

## ⚙️ 5. Cómo Funcionan los Componentes

### A. Ingesta y Estructuración (`pipeline.py`)
1. **Web Scraping:** Emplea `requests` y `BeautifulSoup` para extraer las ofertas bajo el término "tecnología". Cuenta con un **generador temporal sintético de fallback** con datos del mercado panameño (Copa Airlines, Banco General, etc.) que permite que el pipeline corra inmediatamente sin fallar por bloqueos de Cloudflare.
2. **Procesamiento de LLM:** Utiliza la API de Gemini para analizar la oferta cruda y extraer un objeto estructurado JSON con las habilidades técnicas específicas, los salarios máximos/mínimos y la experiencia requerida, guardándolos en la base de datos relacional SQLite (`data/processed/laboral_it.db`) y exportándolos a un archivo CSV consolidado.

### B. Machine Learning (`modelo.py`)
1. **K-Means Clustering:** Convierte las habilidades a una matriz numérica usando TF-IDF. Agrupa las ofertas en 4 clusters correspondientes a perfiles profesionales (e.g. *Frontend / UI Web*, *Data & Analytics*, *DevOps*, *Backend / Core Systems*). Aplica **PCA** para reducir las dimensiones a 2D para graficarlas.
2. **Regresión Lineal (Habilidades Emergentes):** Agrupa las vacantes en periodos quincenales, calcula la frecuencia porcentual de cada tecnología y ajusta una regresión lineal. La **pendiente ($m$)** determina la tasa de crecimiento quincenal de la habilidad, prediciendo cuáles tendrán mayor demanda a futuro.

---

## 🛠️ 6. Ejecución del Pipeline Completo (Opcional)

Si deseas forzar una nueva extracción de datos del scraping y reentrenar los modelos de machine learning, ejecuta en tu terminal con el entorno virtual activo:
```bash
python src/pipeline.py
python src/modelo.py
```
*(Nota: El dashboard de Streamlit también cuenta con un botón en la interfaz para ejecutar este paso automáticamente si deseas ver nuevos datos).*

---

## ❓ 7. Solución de Problemas Comunes (Troubleshooting)

### A. Pantalla "Welcome to Streamlit! Email:" en la terminal
Al ejecutar por primera vez `streamlit run src/app.py`, Streamlit te solicitará un correo electrónico para noticias.
* **Solución**: Simplemente presiona **Enter** en tu teclado (dejando el campo vacío) para continuar. El servidor local se iniciará de inmediato.

### B. Error al activar el entorno virtual en PowerShell (Windows)
Si al ejecutar `.\venv\Scripts\Activate.ps1` recibes un mensaje indicando que *"la ejecución de scripts está deshabilitada en este sistema"*:
* **Solución 1 (CMD)**: Utiliza el **Símbolo del Sistema (CMD)** estándar de Windows en lugar de PowerShell y ejecuta:
  ```cmd
  venv\Scripts\activate
  ```
* **Solución 2 (PowerShell)**: Abre una ventana de PowerShell como Administrador y ejecuta la siguiente política de permisos:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```

### C. Error 404 de API en la Generación de Conclusiones con IA
* **Solución**: Asegúrate de haber guardado tu API Key en el archivo `.env` o en los Secrets de Streamlit Cloud como: `GEMINI_API_KEY = "tu_clave"`. El sistema incluye un bucle de compatibilidad automático que buscará los modelos activos de Google AI Studio en 2026 (`gemini-2.5-flash`, `gemini-pro`, etc.) para evitar caídas del servicio.
