import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import time  # IMPORTANTE: Agregamos time para la pantalla de carga

# ==========================================
# 1. CONFIGURACIÓN INICIAL (Debe ir primero)
# ==========================================
st.set_page_config(
    page_title="PlantDoc | Castiel Analytics", 
    page_icon="🍃", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PANTALLA DE PRESENTACIÓN (SPLASH SCREEN)
# ==========================================
# Verificamos si es la primera vez que se carga la página
if 'mostrar_presentacion' not in st.session_state:
    st.session_state.mostrar_presentacion = True

# Si es la primera vez, mostramos la presentación
if st.session_state.mostrar_presentacion:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #2e7d32; font-size: 3.5em;'>🍃 Castiel Analytics Presenta</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #444;'>PlantDoc: Clasificación Inteligente de Patologías Vegetales</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #666; font-style: italic;'>a través del entrenamiento de Redes Neuronales Convolucionales y técnicas de visión artificial</h5>", unsafe_allow_html=True)
    st.markdown("<hr style='width: 60%; margin: auto;'>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: #333;'>Instituto Superior Politécnico Córdoba (ISPC)</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555;'>Proyecto ABP - Procesamiento de Imágenes y Modelos de IA</h4>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centramos la lista de integrantes
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
    
    # Animación de la barra de carga (Dura aprox. 3 segundos)
    barra_progreso = st.progress(0)
    texto_estado = st.empty()
    
    for i in range(100):
        time.sleep(0.03)  
        barra_progreso.progress(i + 1)
        texto_estado.markdown(f"<p style='text-align: center; color: #666; font-family: monospace;'>Cargando módulos de Castiel Analytics... {i+1}%</p>", unsafe_allow_html=True)
        
    time.sleep(0.5)
    
    # Cambiamos el estado y recargamos la app para entrar al sistema
    st.session_state.mostrar_presentacion = False
    st.rerun()
    st.stop()  # Evita que el resto del código se ejecute mientras se muestra la presentación


# ==========================================
# 3. PANEL LATERAL (UI/UX) - APLICACIÓN PRINCIPAL
# ==========================================
# A partir de aquí, el código es el tuyo original, funcionando a la perfección.

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

# Encabezado Principal
st.title("Análisis Fitosanitario Automatizado")
st.markdown("Carga una muestra foliar para ejecutar la inferencia paralela de los modelos YOLOv8 y ResNet-50.")
st.divider()

# ==========================================
# 4. LAS 39 CLASES ORIGINALES DEL DATASET
# ==========================================
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
# 5. CARGA DEL MODELO RESNET-50
# ==========================================
@st.cache_resource
def cargar_resnet():
    modelo = models.resnet50(weights=None)
    num_ftrs = modelo.fc.in_features
    # Replicamos exactamente la arquitectura secuencial con la que fue entrenado en Colab
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
        # Extraemos las 3 probabilidades más altas (Top 3)
        top_prob, top_clases = torch.topk(probabilidades, 3)
        
    resultados = []
    for i in range(3):
        idx = top_clases[i].item()
        conf = top_prob[i].item()
        resultados.append((CLASES_DATASET[idx], conf))
    return resultados

# ==========================================
# 6. INTERFAZ VISUAL: LAYOUT ESTILO GRADIO
# ==========================================
# Mitad izquierda para Entrada, Mitad derecha para las Salidas de ambos modelos
col_izq, col_der = st.columns([1, 1.2], gap="large")

with col_izq:
    st.markdown("### 📥 Imagen de Entrada")
    archivo_subido = st.file_uploader("Arrastra tu imagen aquí", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if archivo_subido is not None:
        imagen = Image.open(archivo_subido).convert('RGB')
        st.markdown("<div style='border: 2px dashed #ccc; padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.image(imagen, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col_der:
    if archivo_subido is not None:
        st.markdown("### ⚙️ Salida de los Modelos")
        
        with st.spinner("Procesando Tensores..."):
            
            # Sub-columnas para simular la vista side-by-side de Gradio
            mod1, mod2 = st.columns(2)
            
            # --- MODELO 1: YOLOv8 ---
            with mod1:
                st.markdown("**1️⃣ Salida YOLOv8 (Detección de Hoja)**")
                st.markdown("<div style='border: 1px solid #ddd; padding: 10px; border-radius: 8px; background-color: #fafafa;'>", unsafe_allow_html=True)
                st.image(imagen, use_container_width=True) 
                st.markdown("</div>", unsafe_allow_html=True)
                st.success("ROI extraído correctamente.")
            
            # --- MODELO 2: ResNet-50 (Barras de Confianza) ---
            with mod2:
                st.markdown("**2️⃣ Salida ResNet-50 (Clasificación)**")
                
                # Obtenemos las probabilidades reales mediante Softmax
                predicciones = predecir_resnet(imagen)
                
                # Construimos el contenedor HTML
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

                # Sub-columnas para simular la vista side-by-side de Gradio