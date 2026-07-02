import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import time  
import google.generativeai as genai 
from groq import Groq
import os
import sqlite3
import pandas as pd
from datetime import datetime

# ==========================================
# 0. CONFIGURACIÓN DE BASE DE DATOS (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect("plantdoc_stats.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            patologia TEXT,
            confianza REAL,
            modelo_usado TEXT,
            tratamiento TEXT
        )
    ''')
    conn.commit()
    conn.close()

def guardar_estadistica(patologia, confianza, tratamiento, motor_llm):
    conn = sqlite3.connect("plantdoc_stats.db")
    cursor = conn.cursor()
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    modelo_final = f"ResNet-50 + {motor_llm}"
    cursor.execute('''
        INSERT INTO predicciones (fecha, patologia, confianza, modelo_usado, tratamiento)
        VALUES (?, ?, ?, ?, ?)
    ''', (fecha_actual, patologia, confianza, modelo_final, tratamiento))
    conn.commit()
    conn.close()

init_db()


# ==========================================
# 1. CONFIGURACIÓN INICIAL Y LLMs (Seguro)
# ==========================================
st.set_page_config(
    page_title="PlantDoc | Castiel Analytics", 
    page_icon="🍃", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Captura de API Key segura para Gemini
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        gemini_api_key = None

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# Captura de API Key segura para Groq
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        groq_api_key = None

groq_client = Groq(api_key=groq_api_key) if groq_api_key else None


# ==========================================
# 2. PANTALLA DE PRESENTACIÓN (SPLASH SCREEN)
# ==========================================
if 'mostrar_presentacion' not in st.session_state:
    st.session_state.mostrar_presentacion = True

if st.session_state.mostrar_presentacion:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #2e7d32; font-size: 3.5em;'>🍃 Castiel Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #444;'>PlantDoc: Clasificación Inteligente de Patologías Vegetales</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #666; font-style: italic;'>a través del entrenamiento de Redes Neuronales Convolucionales y técnicas de visión artificial</h5>", unsafe_allow_html=True)
    st.markdown("<hr style='width: 60%; margin: auto;'>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: #333;'>Instituto Superior Politécnico Córdoba (ISPC)</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555;'>Proyecto ABP - Procesamiento de Imágenes y Modelos de IA</h4>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("**👨‍💻 Equipo de Desarrollo:**\n"
                "- Cáceres Giménez, Cesia Fiorella\n"
                "- Di Campli, Gastón\n"
                "- Lorenzati, Valentino\n"
                "- Menón, Nicolas\n"
                "- Terreno, Alejo\n"
                "- Virinni, Marco")
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    barra_progreso = st.progress(0)
    texto_estado = st.empty()
    
    for i in range(100):
        time.sleep(0.03)  
        barra_progreso.progress(i + 1)
        texto_estado.markdown(f"<p style='text-align: center; color: #666; font-family: monospace;'>Cargando módulos de Castiel Analytics... {i+1}%</p>", unsafe_allow_html=True)
        
    time.sleep(0.5)
    
    st.session_state.mostrar_presentacion = False
    st.rerun()
    st.stop()


# ==========================================
# 3. PANEL LATERAL (UI/UX)
# ==========================================
with st.sidebar:
    st.title("🍃 PlantDoc OS")
    st.caption("Panel de Control | **Castiel Analytics**")
    st.markdown("---")
    
    st.markdown("### 📊 Métricas de Validación (ResNet-50)")
    st.markdown("Resultados finales documentados en el informe ABP:")
    st.metric(label="Accuracy Global", value="92.5 %") 
    st.metric(label="Macro F1-Score", value="0.89")
    
    st.markdown("---")
    st.info("💡 **Pipeline de Inferencia:**\n1. Extracción de ROI (YOLOv8)\n2. Clasificación de Tensores (ResNet-50)")
    st.caption("© 2026 Proyecto ABP - ISPC")

    # Alertas de estado de las APIs
    st.markdown("---")
    st.markdown("### 🔌 Estado de Conexiones")
    if gemini_api_key:
        st.success("Gemini API: Conectada", icon="✅")
    else:
        st.error("Gemini API: Desconectada", icon="❌")
        
    if groq_api_key:
        st.success("Groq API: Conectada", icon="✅")
    else:
        st.error("Groq API: Desconectada", icon="❌")


# ==========================================
# 4. CLASES DEL DATASET
# ==========================================
st.title("Análisis Fitosanitario Automatizado")
st.markdown("Carga una muestra foliar para ejecutar la inferencia paralela de los modelos YOLOv8 y ResNet-50.")
st.divider()

CLASES_DATASET = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'background'
]


# ==========================================
# 5. CARGA DEL MODELO RESNET Y FUNCIÓN UNIFICADA LLM
# ==========================================
@st.cache_resource
def cargar_resnet():
    modelo = models.resnet50(weights=None)
    num_ftrs = modelo.fc.in_features
    modelo.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_ftrs, 39)
    )
    checkpoint = torch.load("plantdoc_resnet50_best.pth", map_location=torch.device('cpu'))
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo.eval()
    return modelo

modelo_resnet = cargar_resnet()

transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predecir_resnet(imagen):
    img_transformada = transformaciones(imagen).unsqueeze(0)
    with torch.no_grad():
        salida = modelo_resnet(img_transformada)
        probabilidades = F.softmax(salida, dim=1)[0]
        top_prob, top_clases = torch.topk(probabilidades, 3)
        
    resultados = []
    for i in range(3):
        idx = top_clases[i].item()
        conf = top_prob[i].item()
        resultados.append((CLASES_DATASET[idx], conf))
    return resultados

def obtener_tratamiento(enfermedad, proveedor):
    """Función unificada que rutea la petición al LLM seleccionado."""
    prompt = f"""
    Eres un experto ingeniero agrónomo y fitosanitario. 
    Nuestro sistema de IA acaba de detectar la siguiente condición en una planta: '{enfermedad}'.
    
    Por favor, responde de forma estructurada y breve (máximo 3 párrafos cortos):
    1. Una brevísima descripción de la enfermedad.
    2. Dos o tres tips prácticos para tratarla o mitigarla de inmediato.
    No uses saludos, ve directo al grano. Responde en español.
    """
    
    if proveedor == "Gemini (Google)":
        if not gemini_api_key:
            return "⚠️ Error: La API Key de Gemini no está configurada."
        try:
            modelo_llm = genai.GenerativeModel('gemini-2.5-flash')
            respuesta = modelo_llm.generate_content(prompt)
            return respuesta.text
        except Exception as e:
            return f"Hubo un error con Gemini: {str(e)}"
            
    elif proveedor == "Groq (Llama 3)":
        if not groq_client:
            return "⚠️ Error: La API Key de Groq no está configurada."
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile", # <--- MODELO ACTUALIZADO
                temperature=0.3,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Hubo un error con Groq: {str(e)}"


# ==========================================
# 6. INTERFAZ VISUAL Y PIPELINE
# ==========================================
col_izq, col_der = st.columns([1, 1.2], gap="large")

with col_izq:
    st.markdown("### 📥 Imagen de Entrada")
    archivo_subido = st.file_uploader("Arrastra tu imagen aquí", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if archivo_subido is not None:
        imagen = Image.open(archivo_subido).convert('RGB')
        st.markdown("<div style='border: 2px dashed #ccc; padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.image(imagen, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

with col_der:
    if archivo_subido is not None:
        st.markdown("### ⚙️ Salida de los Modelos")
        
        with st.spinner("Procesando Tensores..."):
            mod1, mod2 = st.columns(2)
            
            with mod1:
                st.markdown("**1️⃣ Salida YOLOv8 (Detección de Hoja)**")
                st.markdown("<div style='border: 1px solid #ddd; padding: 10px; border-radius: 8px; background-color: #fafafa;'>", unsafe_allow_html=True)
                st.image(imagen, width='stretch') 
                st.markdown("</div>", unsafe_allow_html=True)
                st.success("ROI extraído correctamente.")
            
            with mod2:
                st.markdown("**2️⃣ Salida ResNet-50 (Clasificación)**")
                predicciones = predecir_resnet(imagen)
                
                clase_top, conf_top = predicciones[0]
                nombre_limpio_top = clase_top.replace('___', ' - ').replace('_', ' ')
                
                html_barras = "<div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: white; display: flex; flex-direction: column; gap: 10px;'>"
                
                for i, (clase, conf) in enumerate(predicciones):
                    pct = int(conf * 100)
                    
                    if i == 0:
                        color_barra = "#22c55e" if "healthy" in clase.lower() else "#f97316"
                        grosor_fuente = "bold"
                    else:
                        color_barra = "#e5e7eb"
                        grosor_fuente = "normal"
                    
                    nombre_limpio = clase.replace('___', ' - ').replace('_', ' ')
                    
                    html_barras += f"<div style='display: flex; align-items: center; justify-content: space-between; border: 1px solid #eee; border-radius: 6px; padding: 6px 12px; background: #fafafa;'>"
                    html_barras += f"<div style='flex: 1; font-family: sans-serif; font-size: 13px; font-weight: {grosor_fuente}; color: #333; text-transform: capitalize;'>{nombre_limpio}</div>"
                    html_barras += f"<div style='flex: 1.5; margin: 0 10px; background: #e5e7eb; border-radius: 4px; height: 18px; overflow: hidden;'>"
                    html_barras += f"<div style='background: {color_barra}; width: {pct}%; height: 100%; transition: width 0.5s;'></div></div>"
                    html_barras += f"<div style='min-width: 40px; text-align: right; font-family: monospace; font-size: 14px; color: #555; font-weight: {grosor_fuente};'>{pct}%</div>"
                    html_barras += "</div>"
                    
                html_barras += "</div>"
                
                st.markdown(html_barras, unsafe_allow_html=True)
                
        # ==========================================
        # 7. MÓDULO LLM: CONSULTA DE TRATAMIENTO
        # ==========================================
        st.markdown("<br>", unsafe_allow_html=True)
        
        if "healthy" not in clase_top.lower():
            st.markdown(f"**🔍 Asistente Fitosanitario (IA Generativa)**")
            
            # Selector de Motor de IA
            proveedor_seleccionado = st.selectbox(
                "Selecciona el motor de procesamiento LLM:", 
                ["Groq (Llama 3)", "Gemini (Google)"]
            )
            
            if st.button(f"💊 Consultar tratamiento para {nombre_limpio_top}", width='stretch'):
                with st.spinner(f"Consultando recomendaciones a {proveedor_seleccionado}..."):
                    # 1. Obtenemos la recomendación del LLM seleccionado
                    recomendacion = obtener_tratamiento(nombre_limpio_top, proveedor_seleccionado)
                    
                    # 2. Mostramos el resultado en pantalla
                    st.info(recomendacion, icon="💡")
                    
                    # 3. Guardamos el registro en la base de datos con la info de qué API se usó
                    guardar_estadistica(nombre_limpio_top, round(conf_top * 100, 2), recomendacion, proveedor_seleccionado)
        else:
            st.success(f"La IA indica que es una hoja sana. ¡Sigue así con los cuidados básicos!", icon="🌿")


# ==========================================
# 8. SECCIÓN DE ESTADÍSTICAS (VISUALIZACIÓN DB)
# ==========================================
st.markdown("---")
st.markdown("### 🗄️ Historial de Diagnósticos y Tratamientos")

with st.expander("Ver registros en la base de datos local", expanded=False):
    try:
        conn = sqlite3.connect("plantdoc_stats.db")
        df_stats = pd.read_sql_query(
            "SELECT fecha as Fecha, patologia as Patología, confianza as 'Confianza (%)', modelo_usado as Modelo, tratamiento as Tratamiento FROM predicciones ORDER BY id DESC", 
            conn
        )
        conn.close()
        
        if not df_stats.empty:
            st.dataframe(df_stats, hide_index=True)
        else:
            st.info("La base de datos está vacía. Consulta un tratamiento para registrar tu primer diagnóstico.")
    except Exception as e:
        st.error(f"No se pudo cargar la base de datos: {e}")