#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
modelo.py - Machine Learning sobre el mercado laboral IT en Panamá.
1. K-Means Clustering (TF-IDF + PCA 2D) con Silhouette Score.
2. Regresión Lineal temporal de habilidades con R² y clasificación de confiabilidad.
Grupo 4 - Gestión de la Información (Semestre I, 2026)
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
CSV_PATH = os.path.join(DATA_PROC_DIR, "vacantes_limpias.csv")
MODEL_JOBS_PATH = os.path.join(DATA_PROC_DIR, "vacantes_modeladas.csv")
MODEL_TRENDS_PATH = os.path.join(DATA_PROC_DIR, "tendencias_skills.csv")


def cargar_datos() -> pd.DataFrame:
    """Carga el CSV limpio generado por el pipeline. Lo ejecuta si no existe."""
    if not os.path.exists(CSV_PATH):
        log.info("CSV no encontrado. Ejecutando pipeline primero...")
        from pipeline import ejecutar_pipeline
        ejecutar_pipeline(num_simulados=200)

    df = pd.read_csv(CSV_PATH)
    df["habilidades"] = df["habilidades"].fillna("")
    median_min = df["salario_min"].median() if not df["salario_min"].isna().all() else 1500.0
    median_max = df["salario_max"].median() if not df["salario_max"].isna().all() else 2500.0
    df["salario_min"] = df["salario_min"].fillna(median_min)
    df["salario_max"] = df["salario_max"].fillna(median_max)
    df["experiencia_anios"] = df["experiencia_anios"].fillna(2)
    return df


# =====================================================================
# 1. Clustering K-Means con Silhouette Score y etiquetas únicas
# =====================================================================

# Posibles etiquetas y sus palabras clave para scoring
_ETIQUETAS_CLUSTER = {
    "Frontend / UI Web": ["react", "html", "css", "angular", "vue", "javascript", "typescript", "ui"],
    "Data & Analytics / BI": ["python", "sql", "power", "bi", "tableau", "excel", "data", "spark", "r"],
    "DevOps & Cloud": ["aws", "docker", "kubernetes", "cloud", "linux", "azure", "terraform", "jenkins"],
    "Backend / Core Systems": ["java", "net", "spring", "node", "postgresql", "mysql", "api", "php", "c#"],
}


def _asignar_etiquetas_unicas(n_clusters: int, order_centroids, terms) -> dict:
    """Asigna etiquetas únicas a cada cluster usando un scoring greedy.
    Garantiza que dos clusters nunca reciban la misma etiqueta.
    """
    # Calcular puntaje de cada cluster para cada etiqueta
    scores: dict[int, dict[str, int]] = {}
    for i in range(n_clusters):
        top_terms = [terms[ind].lower() for ind in order_centroids[i, :10]]
        scores[i] = {}
        for label, keywords in _ETIQUETAS_CLUSTER.items():
            scores[i][label] = sum(1 for kw in keywords if any(kw in t for t in top_terms))

    # Si hay más clusters que etiquetas predefinidas, generar etiquetas genéricas de respaldo
    etiquetas_disponibles = list(_ETIQUETAS_CLUSTER.keys())
    while len(etiquetas_disponibles) < n_clusters:
        etiquetas_disponibles.append(f"Perfil IT #{len(etiquetas_disponibles) + 1}")

    # Asignación greedy: primero asignar el cluster con mejor puntaje total
    cluster_labels: dict[int, str] = {}
    usadas: set[str] = set()

    cluster_order = sorted(range(n_clusters), key=lambda x: max(scores[x].values(), default=0), reverse=True)

    for i in cluster_order:
        disponibles = [l for l in etiquetas_disponibles if l not in usadas]
        mejor = max(disponibles, key=lambda l: scores[i].get(l, 0))
        cluster_labels[i] = mejor
        usadas.add(mejor)

    return cluster_labels


def ejecutar_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    K-Means con TF-IDF sobre habilidades, reducción PCA 2D para visualización.
    Reporta Silhouette Score para justificar la calidad del agrupamiento.
    Las etiquetas de cluster son únicas (sin repetición entre clusters).
    """
    log.info(f"K-Means Clustering con k={n_clusters}...")

    skills_text = df["habilidades"].str.replace(",", " ")
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[\w\.\#\-]+\b')
    tfidf_matrix = vectorizer.fit_transform(skills_text)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(tfidf_matrix)

    # Silhouette Score (métrica de calidad del clustering, rango [-1, 1])
    sil = silhouette_score(tfidf_matrix, df["cluster"], metric="cosine")
    log.info(f"Silhouette Score (cosine): {sil:.4f}  [0.2-0.5 = estructura débil, >0.5 = buena]")
    df["silhouette_score"] = round(float(sil), 4)

    # PCA 2D para visualización
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(tfidf_matrix.toarray())
    df["pca_x"] = pca_coords[:, 0]
    df["pca_y"] = pca_coords[:, 1]

    # Etiquetas únicas por cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    log.info("Top habilidades por cluster:")
    for i in range(n_clusters):
        top = [terms[ind] for ind in order_centroids[i, :5]]
        log.info(f"  Cluster {i}: {', '.join(top)}")

    cluster_labels = _asignar_etiquetas_unicas(n_clusters, order_centroids, terms)
    df["nombre_cluster"] = df["cluster"].map(cluster_labels)

    log.info("Clustering finalizado.\n")
    return df


# =====================================================================
# 2. Regresión Lineal Temporal con R² y manejo de rango corto
# =====================================================================
def predecir_habilidades_emergentes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza la frecuencia temporal de habilidades y ajusta una regresión lineal.
    - Agrupa por semana si el rango de fechas < 90 días (snapshot corto).
    - Agrupa por quincena si el rango ≥ 90 días.
    - Calcula R² de cada regresión; si R² < 0.2, marca la tendencia como 'No confiable'.
    - Si hay muy pocos puntos temporales, reporta honestamente.
    """
    log.info("Análisis de tendencias con Regresión Lineal...")

    df["fecha"] = pd.to_datetime(df["fecha_publicacion"])

    # Desglosar habilidades en filas
    habilidades_desglosadas = []
    for _, row in df.iterrows():
        skills = [s.strip() for s in str(row["habilidades"]).split(",") if s.strip()]
        for skill in skills:
            habilidades_desglosadas.append({
                "id_vacante": row["id"],
                "fecha": row["fecha"],
                "habilidad": skill,
            })

    df_skills = pd.DataFrame(habilidades_desglosadas)
    if df_skills.empty:
        log.warning("Sin habilidades para analizar tendencias.")
        return pd.DataFrame()

    # Determinar granularidad según rango de fechas disponible
    rango_dias = (df_skills["fecha"].max() - df_skills["fecha"].min()).days
    periodo_dias = 7 if rango_dias < 90 else 15
    granularidad = "semanal" if periodo_dias == 7 else "quincenal"
    log.info(f"Rango de fechas: {rango_dias} dias -> agrupacion {granularidad} (periodo={periodo_dias}d).")

    fecha_min = df_skills["fecha"].min()
    df_skills["periodo_id"] = ((df_skills["fecha"] - fecha_min).dt.days // periodo_dias).astype(int)
    df["periodo_id"] = ((df["fecha"] - fecha_min).dt.days // periodo_dias).astype(int)

    vacantes_por_periodo = df.groupby("periodo_id").size().to_dict()
    frecuencia_periodo = df_skills.groupby(["habilidad", "periodo_id"]).size().reset_index(name="conteo")

    def calcular_pct(row):
        total = vacantes_por_periodo.get(row["periodo_id"], 1)
        return (row["conteo"] / total) * 100

    frecuencia_periodo["porcentaje"] = frecuencia_periodo.apply(calcular_pct, axis=1)

    habilidades_populares = (
        df_skills["habilidad"].value_counts()[df_skills["habilidad"].value_counts() >= 5].index.tolist()
    )

    tendencias = []
    max_periodo = int(df_skills["periodo_id"].max())
    n_periodos = max_periodo + 1

    for skill in habilidades_populares:
        datos_skill = frecuencia_periodo[frecuencia_periodo["habilidad"] == skill]

        todos_los_periodos = pd.DataFrame({"periodo_id": range(n_periodos)})
        datos_completos = pd.merge(todos_los_periodos, datos_skill, on="periodo_id", how="left")
        datos_completos["porcentaje"] = datos_completos["porcentaje"].fillna(0.0)
        datos_completos["habilidad"] = skill

        X = datos_completos["periodo_id"].values.reshape(-1, 1)
        y = datos_completos["porcentaje"].values

        model = LinearRegression()
        model.fit(X, y)

        pendiente = float(model.coef_[0])
        intercepto = float(model.intercept_)
        r2 = float(model.score(X, y))

        # Predicción sin warning: usar numpy array con reshape correcto
        proxima_demanda_pred = max(0.0, float(model.predict(np.array([[max_periodo + 1]]))[0]))

        # Clasificar tendencia; si R² < 0.2 la regresión no es confiable
        if n_periodos < 3:
            estado_tendencia = "No confiable / Pocos periodos"
        elif r2 < 0.2:
            estado_tendencia = "No confiable / Pocos datos"
        elif pendiente > 0.3:
            estado_tendencia = "Emergente / Crecimiento Rápido"
        elif pendiente > 0.05:
            estado_tendencia = "Crecimiento Estable"
        elif pendiente < -0.3:
            estado_tendencia = "En Declive"
        else:
            estado_tendencia = "Madura / Estable"

        tendencias.append({
            "habilidad": skill,
            "pendiente": pendiente,
            "intercepto": intercepto,
            "r2": round(r2, 4),
            "porcentaje_actual": float(y[-1]) if len(y) > 0 else 0.0,
            "porcentaje_predicho_futuro": proxima_demanda_pred,
            "tendencia": estado_tendencia,
            "granularidad": granularidad,
            "n_periodos": n_periodos,
        })

    df_tendencias = pd.DataFrame(tendencias).sort_values(by="pendiente", ascending=False)

    log.info("Top 5 Habilidades Emergentes:")
    for _, row in df_tendencias.head(5).iterrows():
        log.info(f"  {row['habilidad']}: pendiente={row['pendiente']:.3f} | R²={row['r2']:.3f} | {row['tendencia']}")

    return df_tendencias


# =====================================================================
# Pipeline principal de ML
# =====================================================================
def ejecutar_modelado():
    """Carga datos, ejecuta K-Means y Regresión Lineal, guarda CSVs de salida."""
    log.info("=" * 70)
    log.info("            MODELADO DE MACHINE LEARNING - GRUPO 4 UTP")
    log.info("=" * 70)

    df = cargar_datos()
    log.info(f"Datos cargados: {len(df)} vacantes.")

    df_clustered = ejecutar_clustering(df, n_clusters=4)
    df_clustered.to_csv(MODEL_JOBS_PATH, index=False, encoding="utf-8-sig")
    log.info(f"Vacantes modeladas guardadas: {MODEL_JOBS_PATH}")

    df_tendencias = predecir_habilidades_emergentes(df_clustered)
    if not df_tendencias.empty:
        df_tendencias.to_csv(MODEL_TRENDS_PATH, index=False, encoding="utf-8-sig")
        log.info(f"Tendencias guardadas: {MODEL_TRENDS_PATH}")

    log.info("Modelado completado exitosamente.")
    log.info("=" * 70)


if __name__ == "__main__":
    ejecutar_modelado()
