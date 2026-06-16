#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline.py - Ingesta y preprocesamiento de datos del mercado IT en Panamá.
Flujo: scraping real (Computrabajo + Konzerta + Arbeitnow API) → LLM (Gemini/heurístico)
       → SQLite + CSV. Si los datos reales son insuficientes (<30), completa con simulados.
Grupo 4 - Gestión de la Información (Semestre I, 2026)
"""

import os
import re
import time
import random
import logging
import sqlite3
import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_PATH = os.path.join(DATA_PROC_DIR, "laboral_it.db")
CSV_PATH = os.path.join(DATA_PROC_DIR, "vacantes_limpias.csv")

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROC_DIR, exist_ok=True)

MIN_REGISTROS_REALES = 30  # umbral antes de activar el generador simulado


# =====================================================================
# 1. Esquema de Datos (Pydantic)
# =====================================================================
class VacanteProcesada(BaseModel):
    puesto: str = Field(description="Nombre normalizado del puesto de trabajo.")
    habilidades_tecnicas: List[str] = Field(description="Lista de habilidades técnicas.")
    salario_min: Optional[float] = Field(None, description="Salario mínimo mensual USD.")
    salario_max: Optional[float] = Field(None, description="Salario máximo mensual USD.")
    experiencia_anios: Optional[int] = Field(None, description="Años de experiencia requeridos.")
    categoria_rol: str = Field(description="Categoría del rol IT.")


# =====================================================================
# 2. Utilidades de Fechas
# =====================================================================
def parse_fecha_relativa(texto: str) -> str:
    """Convierte texto de fecha relativa en español a fecha ISO YYYY-MM-DD.
    Maneja: 'hoy', 'ayer', 'hace N días/semanas/meses', fechas absolutas.
    Si no puede parsear, retorna la fecha de hoy.
    """
    hoy = datetime.date.today()
    if not texto or not texto.strip():
        return hoy.isoformat()

    t = texto.lower().strip()

    if "hoy" in t or "today" in t:
        return hoy.isoformat()
    if "ayer" in t or "yesterday" in t:
        return (hoy - datetime.timedelta(days=1)).isoformat()

    m = re.search(r'hace\s+(\d+)\s*d[ií]as?', t)
    if m:
        return (hoy - datetime.timedelta(days=int(m.group(1)))).isoformat()

    m = re.search(r'hace\s+(\d+)\s*semanas?', t)
    if m:
        return (hoy - datetime.timedelta(weeks=int(m.group(1)))).isoformat()

    m = re.search(r'hace\s+(\d+)\s*mes(?:es)?', t)
    if m:
        return (hoy - datetime.timedelta(days=int(m.group(1)) * 30)).isoformat()

    # Intentar parsear como fecha absoluta
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d %b %Y"):
        try:
            return datetime.datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            pass

    return hoy.isoformat()


# =====================================================================
# 3. Web Scraping con Cloudscraper (bypass Cloudflare)
# =====================================================================
def _get_scraper():
    """Crea sesión cloudscraper. Si no está instalado, usa requests como fallback."""
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        scraper.headers.update({
            "Accept-Language": "es-PA,es;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        return scraper
    except ImportError:
        log.warning("cloudscraper no disponible, usando requests (puede bloquearse).")
        import requests
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "es-PA,es;q=0.9",
        })
        return s


def scrape_portal_computrabajo(query: str = "tecnologia") -> List[Dict[str, Any]]:
    """Scraping de Computrabajo Panamá con cloudscraper y selectores robustos."""
    log.info(f"Intentando Computrabajo: '{query}'...")
    url = f"https://pa.computrabajo.com/trabajo-de-{query}"
    vacantes = []

    try:
        scraper = _get_scraper()
        response = scraper.get(url, timeout=15)
        time.sleep(random.uniform(1.5, 3.0))

        if response.status_code != 200:
            log.warning(f"Computrabajo: status {response.status_code}.")
            return vacantes

        soup = BeautifulSoup(response.text, "html.parser")

        # Probar múltiples selectores en orden de prioridad
        job_cards = (
            soup.find_all("article", class_="box_offer") or
            soup.find_all("article", attrs={"class": re.compile(r"box_offer|job", re.I)}) or
            soup.select("article[data-type='job']") or
            soup.select("[data-qa='job-offer']") or
            soup.find_all("div", class_=re.compile(r"box_offer|offerBox", re.I))
        )

        if not job_cards:
            log.warning("Computrabajo: sin tarjetas con selectores conocidos. Guardando HTML para depuración.")
            raw_path = os.path.join(DATA_RAW_DIR, "computrabajo_raw.html")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response.text[:200000])
            return vacantes

        for card in job_cards[:15]:
            try:
                titulo_el = (
                    card.find("a", class_="js-o-link") or
                    card.select_one("h2 a") or
                    card.find("a", attrs={"data-qa": "job-offer-title"}) or
                    card.find("h2") or
                    card.find("a")
                )
                empresa_el = (
                    card.find("p", class_="mb10") or
                    card.find("a", class_=re.compile(r"company|empresa", re.I)) or
                    card.find("span", class_=re.compile(r"company|empresa", re.I))
                )
                fecha_el = (
                    card.find("time") or
                    card.find("span", class_=re.compile(r"fc_aux|date|fecha", re.I)) or
                    card.find("p", class_=re.compile(r"fs12|date|fecha", re.I))
                )

                titulo = titulo_el.get_text(strip=True) if titulo_el else "Puesto IT"
                empresa = empresa_el.get_text(strip=True) if empresa_el else "Empresa Confidencial"
                empresa = empresa.split("-")[0].strip()

                fecha_texto = ""
                if fecha_el:
                    fecha_texto = fecha_el.get("datetime", "") or fecha_el.get_text(strip=True)
                fecha_pub = parse_fecha_relativa(fecha_texto)

                desc_el = card.find("p", class_=re.compile(r"fs16|description|desc", re.I))
                descripcion = desc_el.get_text(strip=True) if desc_el else f"Oferta de {titulo} en {empresa}, Panamá."

                vacantes.append({
                    "titulo_original": titulo,
                    "empresa": empresa,
                    "descripcion": descripcion,
                    "portal": "Computrabajo",
                    "fecha_publicacion": fecha_pub,
                    "es_simulado": False,
                })
            except Exception as e:
                log.warning(f"Computrabajo - error en tarjeta: {e}")
                continue

        log.info(f"Computrabajo: {len(vacantes)} vacantes extraídas.")
    except Exception as e:
        log.error(f"Error scraping Computrabajo: {e}")

    return vacantes


def scrape_portal_konzerta(query: str = "tecnologia") -> List[Dict[str, Any]]:
    """Scraping de Konzerta Panamá con cloudscraper, múltiples URLs y selectores robustos."""
    log.info(f"Intentando Konzerta: '{query}'...")

    urls_a_intentar = [
        f"https://www.konzerta.com/empleos-busqueda-{query}.html",
        "https://www.konzerta.com/empleos-tecnologia.html",
        f"https://www.konzerta.com/buscar-empleo?q={query}",
    ]

    vacantes = []

    for url in urls_a_intentar:
        try:
            scraper = _get_scraper()
            response = scraper.get(url, timeout=15)
            time.sleep(random.uniform(1.5, 3.0))

            if response.status_code != 200:
                log.warning(f"Konzerta {url}: status {response.status_code}.")
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            job_cards = (
                soup.find_all("div", class_=re.compile(r"job.?card|vacancy|oferta", re.I)) or
                soup.find_all("article", class_=re.compile(r"job|vacante|offer", re.I)) or
                soup.find_all("li", class_=re.compile(r"job|vacante|offer", re.I)) or
                soup.select(".job-item") or
                soup.select("[data-job-id]")
            )

            if not job_cards:
                log.warning(f"Konzerta {url}: sin tarjetas reconocidas. Guardando HTML.")
                raw_path = os.path.join(DATA_RAW_DIR, "konzerta_raw.html")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(response.text[:200000])
                continue

            for card in job_cards[:15]:
                try:
                    titulo_el = (
                        card.find("h2") or
                        card.find("h3") or
                        card.find("a", class_=re.compile(r"title|titulo", re.I))
                    )
                    empresa_el = (
                        card.find("span", class_=re.compile(r"company|empresa", re.I)) or
                        card.find("p", class_=re.compile(r"company|empresa", re.I))
                    )
                    fecha_el = (
                        card.find("time") or
                        card.find("span", class_=re.compile(r"date|fecha", re.I))
                    )

                    titulo = titulo_el.get_text(strip=True) if titulo_el else "Puesto IT"
                    empresa = empresa_el.get_text(strip=True) if empresa_el else "Empresa Confidencial"

                    fecha_texto = ""
                    if fecha_el:
                        fecha_texto = fecha_el.get("datetime", "") or fecha_el.get_text(strip=True)
                    fecha_pub = parse_fecha_relativa(fecha_texto)

                    desc_el = card.find("div", class_=re.compile(r"desc|detail", re.I)) or card.find("p")
                    descripcion = desc_el.get_text(strip=True) if desc_el else f"Vacante de {titulo} en {empresa}."

                    vacantes.append({
                        "titulo_original": titulo,
                        "empresa": empresa,
                        "descripcion": descripcion,
                        "portal": "Konzerta",
                        "fecha_publicacion": fecha_pub,
                        "es_simulado": False,
                    })
                except Exception as e:
                    log.warning(f"Konzerta - error en tarjeta: {e}")
                    continue

            if vacantes:
                log.info(f"Konzerta: {len(vacantes)} vacantes extraídas de {url}.")
                break

        except Exception as e:
            log.error(f"Error scraping Konzerta ({url}): {e}")
            continue

    return vacantes


def fetch_arbeitnow_api() -> List[Dict[str, Any]]:
    """
    Fuente de respaldo con datos REALES vía API pública de Arbeitnow (sin autenticación).
    Complementa los portales panameños con ofertas IT reales (algunas remotas/internacionales)
    para garantizar que el pipeline siempre tenga datos auténticos cuando el scraping falla.
    Fallback secundario: RemoteOK API.
    """
    log.info("Consultando Arbeitnow API (datos reales, sin auth)...")
    vacantes = []
    tech_keywords = {
        "python", "javascript", "java", "react", "node", "sql", "aws", "docker",
        "kubernetes", "data", "backend", "frontend", "fullstack", "devops", "cloud",
        "software", "developer", "engineer", "analyst", "typescript", "flutter", "android",
    }

    try:
        import requests
        resp = requests.get("https://www.arbeitnow.com/api/job-board-api", timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for job in data.get("data", [])[:60]:
            titulo = job.get("title", "")
            if not titulo or not any(kw in titulo.lower() for kw in tech_keywords):
                continue

            descripcion_raw = job.get("description", f"Vacante de {titulo}.")
            desc_limpia = BeautifulSoup(descripcion_raw, "html.parser").get_text(separator=" ", strip=True)[:600]

            fecha_pub = datetime.date.today().isoformat()
            created_at = job.get("created_at")
            if created_at:
                try:
                    fecha_pub = datetime.datetime.fromtimestamp(int(created_at)).date().isoformat()
                except Exception:
                    pass

            vacantes.append({
                "titulo_original": titulo,
                "empresa": job.get("company_name", "Empresa Internacional"),
                "descripcion": desc_limpia,
                "portal": "Arbeitnow API",
                "fecha_publicacion": fecha_pub,
                "es_simulado": False,
            })

        log.info(f"Arbeitnow API: {len(vacantes)} vacantes IT obtenidas.")

    except Exception as e:
        log.error(f"Arbeitnow API falló: {e}. Intentando RemoteOK...")
        try:
            import requests
            resp = requests.get(
                "https://remoteok.com/api",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            jobs_data = resp.json()
            for job in jobs_data[1:40]:
                if not isinstance(job, dict):
                    continue
                titulo = job.get("position", "")
                if not titulo:
                    continue
                tags = job.get("tags", [])
                desc = f"{titulo}. Tecnologías: {', '.join(tags[:6])}." if tags else titulo

                fecha_pub = datetime.date.today().isoformat()
                epoch = job.get("epoch")
                if epoch:
                    try:
                        fecha_pub = datetime.datetime.fromtimestamp(int(epoch)).date().isoformat()
                    except Exception:
                        pass

                vacantes.append({
                    "titulo_original": titulo,
                    "empresa": job.get("company", "Empresa Remota"),
                    "descripcion": desc,
                    "portal": "RemoteOK API",
                    "fecha_publicacion": fecha_pub,
                    "es_simulado": False,
                })
            log.info(f"RemoteOK API (fallback): {len(vacantes)} vacantes.")
        except Exception as e2:
            log.error(f"RemoteOK también falló: {e2}")

    return vacantes


# =====================================================================
# 4. Extracción Inteligente con LLM (Gemini API / Fallback Heurístico)
# =====================================================================
_gemini_quota_agotada = False


def extract_info_with_gemini(titulo: str, descripcion: str) -> VacanteProcesada:
    """
    Usa Gemini para estructurar la vacante. Si la API Key no está configurada
    o falla, usa un parser heurístico local con expresiones regulares.
    """
    global _gemini_quota_agotada
    gemini_key = os.getenv("GEMINI_API_KEY")

    if gemini_key and not _gemini_quota_agotada:
        try:
            import google.generativeai as genai
            import json

            genai.configure(api_key=gemini_key)
            modelos_a_probar = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite-001"]
            response = None
            last_err = None

            prompt = f"""
Analiza la siguiente oferta de empleo y extrae los datos requeridos.

TÍTULO: {titulo}
DESCRIPCIÓN: {descripcion}

Retorna un JSON con exactamente este formato:
{{
    "puesto": "Nombre limpio del puesto",
    "habilidades_tecnicas": ["skill1", "skill2"],
    "salario_min": float o null,
    "salario_max": float o null,
    "experiencia_anios": int o null,
    "categoria_rol": "Una de: Frontend, Backend, Fullstack, Data & Analytics, Mobile, DevOps & Cloud, Soporte & IT, Gestión & Agile"
}}

- Salarios en USD mensual. B/. equivale a USD en Panamá.
- Habilidades técnicas: lenguajes, frameworks, herramientas (no habilidades blandas).
- Experiencia: años requeridos como entero.
"""

            for model_name in modelos_a_probar:
                try:
                    model = genai.GenerativeModel(model_name)
                    gen_config = {"response_mime_type": "application/json"}
                    response = model.generate_content(prompt, generation_config=gen_config)
                    if response and response.text:
                        break
                except Exception as e:
                    last_err = e
                    continue

            if not response:
                raise last_err if last_err else RuntimeError("Sin respuesta de Gemini.")

            data = json.loads(response.text.strip())
            return VacanteProcesada(**data)

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                _gemini_quota_agotada = True
                log.warning("Gemini sin cuota disponible (429). Usando parser heurístico para el resto de la corrida.")
            else:
                log.warning(f"Gemini falló: {e}. Activando parser heurístico...")

    # =====================================================================
    # FALLBACK: Heurísticas NLP local
    # =====================================================================
    texto_completo = f"{titulo} {descripcion}".lower()

    diccionario_skills = [
        "python", "javascript", "typescript", "react", "angular", "vue", "node.js", "java", "spring boot",
        "c#", ".net", "php", "laravel", "sql", "postgresql", "mysql", "oracle", "mongodb", "aws", "azure",
        "docker", "kubernetes", "git", "power bi", "tableau", "excel", "r", "spark", "hadoop", "c++", "go",
        "flutter", "react native", "swift", "kotlin", "html", "css", "sass", "scrum", "agile", "jira", "linux",
        "terraform", "jenkins", "redis", "elasticsearch",
    ]
    nombres_bonitos = {
        "python": "Python", "javascript": "JavaScript", "typescript": "TypeScript",
        "react": "React", "angular": "Angular", "vue": "Vue", "node.js": "Node.js",
        "java": "Java", "spring boot": "Spring Boot", "c#": "C#", ".net": ".NET",
        "php": "PHP", "laravel": "Laravel", "sql": "SQL", "postgresql": "PostgreSQL",
        "mysql": "MySQL", "oracle": "Oracle", "mongodb": "MongoDB", "aws": "AWS",
        "azure": "Azure", "docker": "Docker", "kubernetes": "Kubernetes", "git": "Git",
        "power bi": "Power BI", "tableau": "Tableau", "excel": "Excel", "r": "R",
        "spark": "Spark", "hadoop": "Hadoop", "c++": "C++", "go": "Go",
        "flutter": "Flutter", "react native": "React Native", "swift": "Swift",
        "kotlin": "Kotlin", "html": "HTML", "css": "CSS", "sass": "Sass",
        "scrum": "Scrum", "agile": "Agile", "jira": "Jira", "linux": "Linux",
        "terraform": "Terraform", "jenkins": "Jenkins", "redis": "Redis",
        "elasticsearch": "Elasticsearch",
    }

    skills_encontradas = []
    for skill in diccionario_skills:
        patron = r'\b' + re.escape(skill) + r'\b' if skill not in ("c#", ".net") else re.escape(skill)
        if re.search(patron, texto_completo):
            skills_encontradas.append(nombres_bonitos.get(skill, skill.capitalize()))

    if not skills_encontradas:
        skills_encontradas = ["Excel", "SQL"]

    exp_matches = re.findall(r'(\d+)\s*(?:años?|years?)\s*(?:de\s*experiencia)?', texto_completo)
    exp = int(exp_matches[0]) if exp_matches else random.randint(1, 3)

    salario_min, salario_max = None, None
    salario_matches = re.findall(
        r'(?:usd|\$|b/\.?)\s*(\d+[,.]?\d*)\s*(?:a|-)\s*(?:usd|\$|b/\.?)\s*(\d+[,.]?\d*)',
        texto_completo,
    )
    if salario_matches:
        try:
            salario_min = float(salario_matches[0][0].replace(",", ""))
            salario_max = float(salario_matches[0][1].replace(",", ""))
        except Exception as e:
            log.debug(f"No se pudo parsear el rango salarial '{salario_matches[0]}': {e}")
    else:
        salarios_ind = re.findall(
            r'(?:salario|sueldo|pago)\s*(?:de|hasta)?\s*(?:usd|\$|b/\.?)\s*(\d+[,.]?\d*)',
            texto_completo,
        )
        if salarios_ind:
            try:
                base_val = float(salarios_ind[0].replace(",", ""))
                if 400 < base_val < 10000:
                    salario_min = base_val * 0.9
                    salario_max = base_val * 1.1
            except Exception as e:
                log.debug(f"No se pudo parsear el salario individual '{salarios_ind[0]}': {e}")

    categoria = "Backend"
    puesto_limpio = titulo
    for cat, keywords in {
        "Frontend": ["frontend", "front", "react", "angular", "vue", "html", "css", "ui"],
        "Backend": ["backend", "back", "java", "python", "php", "c#", ".net", "node", "api", "spring"],
        "Fullstack": ["fullstack", "full-stack", "full stack", "desarrollador integral"],
        "Data & Analytics": ["data", "datos", "analista de datos", "bi", "power bi", "tableau", "cienc", "analytics"],
        "Mobile": ["mobile", "android", "ios", "flutter", "react native", "swift", "kotlin"],
        "DevOps & Cloud": ["devops", "cloud", "aws", "azure", "docker", "kubernetes", "infraestructura", "sysadmin"],
        "Soporte & IT": ["soporte", "support", "redes", "networking", "helpdesk", "técnico", "mantenimiento"],
        "Gestión & Agile": ["scrum", "product owner", "project manager", "agile", "gestor", "coordinador"],
    }.items():
        if any(kw in texto_completo for kw in keywords):
            categoria = cat
            if "data" in texto_completo:
                puesto_limpio = "Data Analyst / Scientist"
            elif "frontend" in texto_completo:
                puesto_limpio = "Frontend Developer"
            elif "backend" in texto_completo:
                puesto_limpio = "Backend Developer"
            elif "fullstack" in texto_completo:
                puesto_limpio = "Fullstack Developer"
            elif "devops" in texto_completo:
                puesto_limpio = "DevOps Engineer"
            elif "soporte" in texto_completo:
                puesto_limpio = "Soporte Técnico IT"
            else:
                puesto_limpio = titulo
            break

    return VacanteProcesada(
        puesto=puesto_limpio,
        habilidades_tecnicas=skills_encontradas,
        salario_min=salario_min,
        salario_max=salario_max,
        experiencia_anios=exp,
        categoria_rol=categoria,
    )


# =====================================================================
# 5. Generador de Datos Simulados (respaldo sin sesgo temporal)
# =====================================================================
def generate_panama_mock_data(num_records: int = 150) -> List[Dict[str, Any]]:
    """
    Genera dataset sintético del mercado IT panameño.
    Las habilidades se asignan aleatoriamente sin sesgo temporal artificial,
    para no contaminar el análisis de tendencias de regresión lineal.
    Todos los registros llevan es_simulado=True.
    """
    log.info(f"Generando {num_records} vacantes simuladas (respaldo)...")

    empresas_pa = [
        "Banco General", "Copa Airlines", "Telered", "Autoridad del Canal de Panamá (ACP)",
        "Global Bank", "Dell Technologies Panamá", "Tigo Panamá", "Cable & Wireless",
        "Multibank", "Banistmo", "Sonda Panamá", "Caja de Seguro Social",
        "Encuentra24 PA", "KPMG Panamá", "EY Panamá", "PwC Panamá", "GBM Panamá",
    ]

    portales = ["Konzerta", "Computrabajo", "LinkedIn"]

    tecnologias_pool = {
        "Frontend": ["React", "JavaScript", "HTML", "CSS", "TypeScript", "Angular", "Vue", "Git"],
        "Backend": ["Python", "Java", "C#", ".NET", "SQL", "Spring Boot", "Node.js", "PostgreSQL", "Git", "Docker"],
        "Fullstack": ["React", "Node.js", "JavaScript", "SQL", "Python", "Git", "Docker", "AWS"],
        "Data & Analytics": ["Python", "SQL", "Power BI", "Tableau", "Excel", "R", "Spark", "PostgreSQL"],
        "Mobile": ["Flutter", "Kotlin", "Swift", "React Native", "JavaScript", "Git"],
        "DevOps & Cloud": ["AWS", "Docker", "Kubernetes", "Linux", "Git", "Azure", "Terraform", "Python"],
        "Soporte & IT": ["Linux", "Excel", "Windows Server", "Cisco", "Jira"],
        "Gestión & Agile": ["Scrum", "Agile", "Jira", "Excel"],
    }

    roles_por_cat = {
        "Frontend": ["Frontend Developer", "React Developer", "UI Developer"],
        "Backend": ["Backend Developer", "Java Engineer", "Python Developer", ".NET Consultant"],
        "Fullstack": ["Fullstack Engineer", "Desarrollador Web Fullstack"],
        "Data & Analytics": ["Data Analyst", "Data Scientist", "BI Engineer", "Analista de Datos"],
        "Mobile": ["Mobile Developer", "iOS App Developer", "Android Developer"],
        "DevOps & Cloud": ["DevOps Engineer", "Cloud Infrastructure Specialist", "Administrador Cloud"],
        "Soporte & IT": ["Soporte Técnico IT", "Administrador de Sistemas", "Ingeniero de Soporte"],
        "Gestión & Agile": ["Scrum Master", "Product Owner", "IT Project Manager"],
    }

    salarios_por_cat = {
        "Frontend": (1200, 2800),
        "Backend": (1400, 3500),
        "Fullstack": (1600, 4000),
        "Data & Analytics": (1500, 3800),
        "Mobile": (1300, 3000),
        "DevOps & Cloud": (1800, 4500),
        "Soporte & IT": (800, 1800),
        "Gestión & Agile": (1800, 4200),
    }

    hoy = datetime.date.today()
    datos = []

    for _ in range(num_records):
        cat = random.choice(list(tecnologias_pool.keys()))
        puesto = random.choice(roles_por_cat[cat])
        empresa = random.choice(empresas_pa)
        portal = random.choice(portales)

        dias_atras = random.randint(0, 180)
        fecha_pub = hoy - datetime.timedelta(days=dias_atras)

        habilidades_posibles = tecnologias_pool[cat]
        # Asignación aleatoria sin sesgo temporal: la distribución es uniforme
        skills = list(set(random.sample(habilidades_posibles, k=random.randint(2, min(5, len(habilidades_posibles))))))

        rango = salarios_por_cat[cat]
        sal_min = round(random.uniform(rango[0], rango[0] + (rango[1] - rango[0]) * 0.4), -2)
        sal_max = round(random.uniform(sal_min + 300, rango[1]), -2)
        exp = random.randint(4, 8) if sal_min > 2500 else random.randint(1, 3)

        descripciones = [
            f"Buscamos un {puesto} para nuestro equipo de tecnología en {empresa}, Panamá.",
            f"En {empresa} requerimos {puesto} con experiencia en {', '.join(skills[:3])} para liderar nuestra transformación digital.",
            f"Gran oportunidad en Ciudad de Panamá. {empresa} busca {puesto} con al menos {exp} años de experiencia.",
        ]

        datos.append({
            "titulo_original": puesto,
            "empresa": empresa,
            "descripcion": random.choice(descripciones),
            "portal": portal,
            "fecha_publicacion": fecha_pub.isoformat(),
            "puesto": puesto,
            "habilidades_tecnicas": skills,
            "salario_min": float(sal_min),
            "salario_max": float(sal_max),
            "experiencia_anios": exp,
            "categoria_rol": cat,
            "es_simulado": True,
        })

    return datos


# =====================================================================
# 6. Persistencia en SQLite y CSV
# =====================================================================
def guardar_en_db(vacantes: List[Dict[str, Any]]):
    """Guarda vacantes en SQLite. Recrea las tablas en cada ejecución para
    garantizar un esquema limpio y evitar acumulación de datos duplicados."""
    log.info(f"Guardando {len(vacantes)} vacantes en SQLite ({DB_PATH})...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Recrear tablas con esquema actualizado (incluye es_simulado)
    cursor.executescript("""
    DROP TABLE IF EXISTS vacante_habilidad;
    DROP TABLE IF EXISTS habilidades;
    DROP TABLE IF EXISTS vacantes;

    CREATE TABLE vacantes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        titulo_original TEXT,
        puesto TEXT,
        empresa TEXT,
        portal TEXT,
        fecha_publicacion DATE,
        salario_min REAL,
        salario_max REAL,
        experiencia_anios INTEGER,
        categoria_rol TEXT,
        descripcion TEXT,
        es_simulado INTEGER DEFAULT 0
    );

    CREATE TABLE habilidades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT UNIQUE
    );

    CREATE TABLE vacante_habilidad (
        vacante_id INTEGER,
        habilidad_id INTEGER,
        PRIMARY KEY (vacante_id, habilidad_id),
        FOREIGN KEY (vacante_id) REFERENCES vacantes (id) ON DELETE CASCADE,
        FOREIGN KEY (habilidad_id) REFERENCES habilidades (id) ON DELETE CASCADE
    );
    """)

    for vac in vacantes:
        cursor.execute("""
        INSERT INTO vacantes (
            titulo_original, puesto, empresa, portal, fecha_publicacion,
            salario_min, salario_max, experiencia_anios, categoria_rol, descripcion, es_simulado
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vac.get("titulo_original"),
            vac.get("puesto"),
            vac.get("empresa"),
            vac.get("portal"),
            vac.get("fecha_publicacion"),
            vac.get("salario_min"),
            vac.get("salario_max"),
            vac.get("experiencia_anios"),
            vac.get("categoria_rol"),
            vac.get("descripcion"),
            int(bool(vac.get("es_simulado", False))),
        ))

        vacante_id = cursor.lastrowid

        for skill in vac.get("habilidades_tecnicas", []):
            cursor.execute("INSERT OR IGNORE INTO habilidades (nombre) VALUES (?)", (skill,))
            cursor.execute("SELECT id FROM habilidades WHERE nombre = ?", (skill,))
            habilidad_id = cursor.fetchone()[0]
            cursor.execute("INSERT OR IGNORE INTO vacante_habilidad VALUES (?, ?)", (vacante_id, habilidad_id))

    conn.commit()
    conn.close()
    log.info("Datos guardados en SQLite.")


def exportar_a_csv():
    """Une tablas y exporta CSV desnormalizado para Pandas/ML."""
    log.info(f"Exportando CSV a {CSV_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    df_vacantes = pd.read_sql_query("SELECT * FROM vacantes", conn)

    query_skills = """
    SELECT vh.vacante_id, GROUP_CONCAT(h.nombre, ',') as habilidades
    FROM vacante_habilidad vh
    JOIN habilidades h ON vh.habilidad_id = h.id
    GROUP BY vh.vacante_id
    """
    df_skills = pd.read_sql_query(query_skills, conn)

    df_final = pd.merge(df_vacantes, df_skills, left_on="id", right_on="vacante_id", how="left")
    df_final.drop(columns=["vacante_id"], inplace=True, errors="ignore")
    df_final.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    conn.close()
    log.info(f"CSV exportado: {len(df_final)} filas.")


# =====================================================================
# 7. Pipeline Principal
# =====================================================================
def ejecutar_pipeline(num_simulados: int = 150):
    """
    Pipeline completo: primero intenta datos REALES (scraping + API pública).
    Solo si obtiene menos de MIN_REGISTROS_REALES registros reales, completa
    con el generador simulado como respaldo. Todos los registros tienen la
    columna es_simulado indicando su origen.
    """
    log.info("=" * 70)
    log.info("         PIPELINE DE MERCADO LABORAL IT - GRUPO 4 UTP")
    log.info("=" * 70)

    # --- FASE 1: Scraping real ---
    vacantes_crudas = []

    log.info("Fase 1: Intentando scraping real de portales...")
    vacantes_crudas.extend(scrape_portal_computrabajo())
    vacantes_crudas.extend(scrape_portal_konzerta())

    # --- FASE 2: API pública como respaldo real ---
    log.info("Fase 2: Consultando API pública de empleos IT...")
    vacantes_crudas.extend(fetch_arbeitnow_api())

    registros_reales = len(vacantes_crudas)
    log.info(f"Total de registros reales obtenidos: {registros_reales}")

    # --- FASE 3: Estructurar vacantes reales con LLM ---
    vacantes_procesadas = []
    if vacantes_crudas:
        log.info(f"Fase 3: Procesando {registros_reales} vacantes reales con LLM/heurístico...")
        for i, vac in enumerate(vacantes_crudas):
            log.info(f"  [{i+1}/{registros_reales}] {vac['titulo_original'][:60]}")
            info = extract_info_with_gemini(vac["titulo_original"], vac["descripcion"])
            vacantes_procesadas.append({
                "titulo_original": vac["titulo_original"],
                "empresa": vac["empresa"],
                "portal": vac["portal"],
                "fecha_publicacion": vac["fecha_publicacion"],
                "descripcion": vac["descripcion"],
                "puesto": info.puesto,
                "habilidades_tecnicas": info.habilidades_tecnicas,
                "salario_min": info.salario_min,
                "salario_max": info.salario_max,
                "experiencia_anios": info.experiencia_anios,
                "categoria_rol": info.categoria_rol,
                "es_simulado": False,
            })

    # --- FASE 4: Completar con simulados si faltan datos ---
    if registros_reales < MIN_REGISTROS_REALES:
        faltantes = max(num_simulados - len(vacantes_procesadas), 1)
        log.info(
            f"Fase 4: Solo {registros_reales} registros reales (mínimo={MIN_REGISTROS_REALES}). "
            f"Completando con {faltantes} simulados..."
        )
        vacantes_procesadas.extend(generate_panama_mock_data(num_records=faltantes))
    else:
        log.info(f"Suficientes datos reales ({registros_reales}). No se usarán simulados.")

    # --- FASE 5: Guardar ---
    reales = sum(1 for v in vacantes_procesadas if not v.get("es_simulado"))
    simulados = sum(1 for v in vacantes_procesadas if v.get("es_simulado"))
    log.info(f"Total a guardar: {len(vacantes_procesadas)} ({reales} reales / {simulados} simulados)")

    guardar_en_db(vacantes_procesadas)
    exportar_a_csv()

    log.info("Pipeline completado. Listo para el modelado de ML.")
    log.info("=" * 70)
    return reales, simulados


if __name__ == "__main__":
    ejecutar_pipeline(num_simulados=200)
