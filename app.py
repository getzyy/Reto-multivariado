import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA (Debe ser la primera instrucción de Streamlit)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Recomendador de Grasas Inteligente",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        box-shadow: 0px 2px 2px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #0068c9;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA DE DATOS (Cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    try:
        # Asegúrate de que estos nombres coincidan con tus archivos reales
        df_base = pd.read_csv("final_df_final.csv")
        df_anderol = pd.read_csv("competencia/Anderol.csv")
        df_solgelsal = pd.read_csv("competencia/Colgelsa - Hoja 1.csv")
        df_starplex = pd.read_csv("competencia/Starplex.csv")
    except FileNotFoundError:
        st.error("Error crítico: No se encuentran los archivos CSV. Verifica las rutas en la carpeta.")
        return None, None

    def limpiar_columnas(df):
        df = df.copy()
        df.columns = (df.columns.str.replace("âˆž", "°", regex=False)
                      .str.replace("∞", "°", regex=False)
                      .str.replace("Ã›", "ó", regex=False)
                      .str.replace("Û", "ó", regex=False)
                      .str.replace("Â·", "á", regex=False)
                      .str.replace("·", "á", regex=False)
                      .str.replace("Ò", "ñ", regex=False)
                      .str.replace("µ", "u", regex=False)
                      .str.strip())
        return df

    # Limpieza
    df_base = limpiar_columnas(df_base)
    df_anderol = limpiar_columnas(df_anderol)
    df_solgelsal = limpiar_columnas(df_solgelsal)
    df_starplex = limpiar_columnas(df_starplex)

    columnas_modelo = [
        "Aceite Base", "Espesante", "Grado NLGI Consistencia",
        "Viscosidad del Aceite Base a 40°C. cSt", "Penetración de Cono a 25°C, 0.1mm",
        "Punto de Gota, °C", "Estabilidad Mecánica, %",
        "Punto de Soldadura Cuatro Bolas, kgf", "Desgaste Cuatro Bolas, mm",
        "Carga Timken Ok, lb", "Resistencia al Lavado por Agua a 80°C, %",
        "Factor de Velocidad", "Temperatura de Servicio °C, min",
        "Temperatura de Servicio °C, max", "descripcion", "color", "textura"
    ]

    # Filtrado y Etiquetado
    df_base = df_base[columnas_modelo].copy()
    df_anderol = df_anderol[columnas_modelo].copy()
    df_solgelsal = df_solgelsal[columnas_modelo].copy()
    df_starplex = df_starplex[columnas_modelo].copy()

    df_base["Empresa"] = "Interlub"
    df_anderol["Empresa"] = "Anderol"
    df_solgelsal["Empresa"] = "Solgelsal"
    df_starplex["Empresa"] = "Starplex"

    # Unión
    df_total = pd.concat([df_base, df_anderol, df_solgelsal, df_starplex], ignore_index=True)
    df_total = df_total.reset_index(drop=True)

    # Crear columnas auxiliares
    df_total["nombre_legible"] = (
        df_total["Empresa"].astype(str) + " | " +
        df_total["Aceite Base"].astype(str) + " | NLGI " +
        df_total["Grado NLGI Consistencia"].astype(str)
    )

    df_total["soup"] = (
        df_total["Aceite Base"].astype(str) + " " +
        df_total["Espesante"].astype(str) + " " +
        df_total["color"].astype(str) + " " +
        df_total["textura"].astype(str) + " " +
        df_total["descripcion"].astype(str)
    )

    return df_total, columnas_modelo

# -----------------------------------------------------------------------------
# 2. PROCESAMIENTO MATRICES Y MODELOS (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def compute_similarity_models(df_total):
    feature_cols = [
        "Grado NLGI Consistencia", "Viscosidad del Aceite Base a 40°C. cSt",
        "Penetración de Cono a 25°C, 0.1mm", "Punto de Gota, °C",
        "Estabilidad Mecánica, %", "Punto de Soldadura Cuatro Bolas, kgf",
        "Desgaste Cuatro Bolas, mm", "Carga Timken Ok, lb",
        "Resistencia al Lavado por Agua a 80°C, %", "Factor de Velocidad",
        "Temperatura de Servicio °C, min", "Temperatura de Servicio °C, max",
    ]

    # --- Numérico ---
    X_num = df_total[feature_cols].copy()
    for col in feature_cols:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    
    means = X_num.mean()
    X_num = X_num.fillna(means)
    
    scaler_num = StandardScaler()
    X_num_scaled = scaler_num.fit_transform(X_num)
    sim_num = cosine_similarity(X_num_scaled, X_num_scaled)

    # --- Metadata ---
    count_vec = CountVectorizer(stop_words=None)
    count_matrix = count_vec.fit_transform(df_total["soup"])
    sim_meta = cosine_similarity(count_matrix, count_matrix)

    # --- Normalización ---
    scaler_sim_num = MinMaxScaler()
    sim_num_norm = scaler_sim_num.fit_transform(sim_num)

    scaler_sim_meta = MinMaxScaler()
    sim_meta_norm = scaler_sim_meta.fit_transform(sim_meta)
    
    models = {
        "scaler_num": scaler_num,
        "count_vec": count_vec,
        "feature_cols": feature_cols,
        "means": means,
        "X_num_scaled": X_num_scaled,
        "count_matrix": count_matrix,
        "sim_num_norm": sim_num_norm,
        "sim_meta_norm": sim_meta_norm,
        "scaler_sim_num": scaler_sim_num, 
        "scaler_sim_meta": scaler_sim_meta 
    }
    
    return models, X_num

# -----------------------------------------------------------------------------
# 3. LÓGICA DE RECOMENDACIÓN
# -----------------------------------------------------------------------------

def get_recommendations_existing(idx_input, df, models, alpha, filter_competitors=False):
    sim_num = models["sim_num_norm"]
    sim_meta = models["sim_meta_norm"]
    
    hybrid_row = alpha * sim_num[idx_input] + (1 - alpha) * sim_meta[idx_input]
    
    sim_scores = list(enumerate(hybrid_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx_input]

    indices = [s[0] for s in sim_scores]
    scores = [s[1] for s in sim_scores]

    df_results = df.iloc[indices].copy()
    df_results["Similitud"] = scores

    if filter_competitors:
        # Solo mostrar productos de Interlub
        df_results = df_results[df_results["Empresa"] == "Interlub"]
    
    return df_results.head(5)

def get_recommendations_from_attributes(user_input, df_total, models, alpha):
    # 1. Numérico
    input_vector = models["means"].copy()
    
    filled_count = 0
    for col, val in user_input["numeric"].items():
        if val is not None and val != 0: 
            input_vector[col] = val
            filled_count += 1
            
    if filled_count < 2:
        return None, "Por favor ingresa al menos 2 valores numéricos para poder calcular similitud."

    input_vector_reshaped = input_vector.values.reshape(1, -1)
    input_scaled = models["scaler_num"].transform(input_vector_reshaped)
    
    sim_num_new = cosine_similarity(input_scaled, models["X_num_scaled"]).flatten()
    sim_num_new_norm = (sim_num_new - sim_num_new.min()) / (sim_num_new.max() - sim_num_new.min() + 1e-9)

    # 2. Texto
    user_soup = user_input.get("soup", "")
    if user_soup.strip() == "":
        sim_meta_new_norm = np.zeros(len(df_total))
    else:
        input_count = models["count_vec"].transform([user_soup])
        sim_meta_new = cosine_similarity(input_count, models["count_matrix"]).flatten()
        sim_meta_new_norm = (sim_meta_new - sim_meta_new.min()) / (sim_meta_new.max() - sim_meta_new.min() + 1e-9)

    # 3. Híbrido
    hybrid_scores = alpha * sim_num_new_norm + (1 - alpha) * sim_meta_new_norm
    
    sim_scores = list(enumerate(hybrid_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    indices = [s[0] for s in sim_scores]
    scores = [s[1] for s in sim_scores]
    
    df_results = df_total.iloc[indices].copy()
    df_results["Similitud"] = scores
    
    return df_results.head(5), None

# -----------------------------------------------------------------------------
# 4. VISUALIZACIÓN (Radar Chart)
# -----------------------------------------------------------------------------
def plot_radar_chart(df_raw, row_data_dict, idx_recomendado, nombre_rec):
    cols_radar = ["Grado NLGI Consistencia", "Punto de Gota, °C", 
                  "Punto de Soldadura Cuatro Bolas, kgf", "Temperatura de Servicio °C, max"]
    
    # Datos Usuario 
    vals_user = []
    for c in cols_radar:
        if isinstance(row_data_dict, pd.Series): 
            vals_user.append(row_data_dict[c])
        else:
            val = row_data_dict.get(c, 0)
            vals_user.append(val if val is not None else 0)

    vals_user = np.array(vals_user)
    vals_rec = df_raw.iloc[idx_recomendado][cols_radar].fillna(0).values

    # Escalar visualmente
    combined = np.vstack([vals_user, vals_rec])
    max_vals = combined.max(axis=0)
    max_vals[max_vals == 0] = 1 
    
    vals_user_norm = vals_user / max_vals
    vals_rec_norm = vals_rec / max_vals

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_user_norm, theta=cols_radar, fill='toself', name='Selección/Input', line_color='blue'))
    fig.add_trace(go.Scatterpolar(r=vals_rec_norm, theta=cols_radar, fill='toself', name='Recomendada', line_color='green'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=300, margin=dict(l=40, r=40, t=20, b=20))
    return fig

# -----------------------------------------------------------------------------
# 5. UI PRINCIPAL (Función MAIN corregida con IDs ÚNICOS)
# -----------------------------------------------------------------------------
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2082/2082986.png", width=80)
    st.sidebar.title("Configuración")
    
    with st.spinner("Inicializando motor de IA..."):
        df_total, _ = load_and_clean_data()
        if df_total is None: return
        models, df_numeric_raw = compute_similarity_models(df_total)

    alpha = st.sidebar.slider("Balance IA (Alpha)", 0.0, 1.0, 0.6, 0.05)
    st.sidebar.info("**Alpha 0.6:** Balance recomendado entre química y datos técnicos.")

    st.title("🔬 Sistema de Recomendación de Grasas")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs([
        "🏢 Análisis Interno", 
        "⚔️ Benchmarking", 
        "🛠️ Búsqueda por Atributos"
    ])

    # --- Función para renderizar Tabs 1 y 2 con KEY ÚNICA ---
    def render_analysis(mode_internal, unique_key_id):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Grasa de Referencia")
            opciones = df_total["nombre_legible"].tolist()
            
            # Key única para el selectbox
            seleccion = st.selectbox(
                "Selecciona grasa base:", 
                opciones, 
                key=f"select_{unique_key_id}"
            )
            idx_sel = df_total[df_total["nombre_legible"] == seleccion].index[0]
            st.info(f"Analizando: {seleccion}")

        with col2:
            st.subheader("Resultados")
            results = get_recommendations_existing(idx_sel, df_total, models, alpha, filter_competitors=mode_internal)
            
            if not results.empty:
                top = results.iloc[0]
                idx_best = results.index[0]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Top Match", top["Empresa"])
                c2.metric("Similitud", f"{top['Similitud']:.1%}")
                c3.metric("Punto Gota", f"{top['Punto de Gota, °C']} °C")
                
                fig = plot_radar_chart(df_numeric_raw, df_total.iloc[idx_sel], idx_best, top["nombre_legible"])
                
                # ---> AQUÍ ESTÁ LA SOLUCIÓN DEFINITIVA <---
                # Usamos el string único pasado como argumento
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{unique_key_id}")
                
                st.dataframe(
                    results[["Empresa", "nombre_legible", "Similitud", "Aceite Base", "Espesante"]].style.background_gradient(subset=["Similitud"], cmap="Greens"), 
                    use_container_width=True
                )

    # Llamamos a la función pasando strings explícitos como ID
    with tab1: 
        render_analysis(mode_internal=True, unique_key_id="interno_tab")
    
    with tab2: 
        render_analysis(mode_internal=False, unique_key_id="competencia_tab")

    # --- Pestaña 3: MANUAL ---
    with tab3:
        st.markdown("#### 🛠️ Diseña tu grasa ideal")
        st.caption("Ingresa los valores técnicos que necesitas (mínimo 2 campos numéricos) y la IA buscará la grasa más parecida en la base de datos.")
        
        with st.form("form_atributos"):
            col_a, col_b, col_c = st.columns(3)
            val_nlgi = col_a.number_input("Grado NLGI", min_value=0.0, max_value=3.0, step=0.5, value=0.0)
            val_gota = col_b.number_input("Punto de Gota (°C)", min_value=0.0, value=0.0)
            val_4bolas = col_c.number_input("Carga 4 Bolas (kgf)", min_value=0.0, value=0.0)
            
            col_d, col_e, col_f = st.columns(3)
            val_visc = col_d.number_input("Viscosidad 40°C (cSt)", min_value=0.0, value=0.0)
            val_temp_max = col_e.number_input("Temp. Servicio Max (°C)", min_value=0.0, value=0.0)
            val_factor = col_f.number_input("Factor Velocidad", min_value=0.0, value=0.0)

            st.markdown("---")
            val_texto = st.text_input("Palabras clave / Composición (Opcional)", placeholder="Ej: Litio complejo sintética")
            
            submitted = st.form_submit_button("🔍 Buscar Grasa Similar")
            
            if submitted:
                user_dict = {
                    "numeric": {
                        "Grado NLGI Consistencia": val_nlgi if val_nlgi > 0 else None,
                        "Punto de Gota, °C": val_gota if val_gota > 0 else None,
                        "Punto de Soldadura Cuatro Bolas, kgf": val_4bolas if val_4bolas > 0 else None,
                        "Viscosidad del Aceite Base a 40°C. cSt": val_visc if val_visc > 0 else None,
                        "Temperatura de Servicio °C, max": val_temp_max if val_temp_max > 0 else None,
                        "Factor de Velocidad": val_factor if val_factor > 0 else None
                    },
                    "soup": val_texto
                }
                
                results, error = get_recommendations_from_attributes(user_dict, df_total, models, alpha)
                if results is None:
                    return
            
                if error:
                    st.error(error)
                else:
                    st.success("¡Análisis completado!")
                    top = results.iloc[0]
                    idx_best = results.index[0]
                    
                    mc1, mc2 = st.columns([1, 2])
                    with mc1:
                        st.metric("Mejor Opción", top["nombre_legible"])
                        st.metric("Confianza", f"{top['Similitud']:.2%}")
                    
                    with mc2:
                        flat_user_data = user_dict["numeric"]
                        fig = plot_radar_chart(df_numeric_raw, flat_user_data, idx_best, top["nombre_legible"])
                        # Key explícita para el tercer gráfico
                        st.plotly_chart(fig, use_container_width=True, key="chart_manual_search")
                    
                    st.dataframe(
                        results[["Empresa", "nombre_legible", "Similitud", "Grado NLGI Consistencia"]].style.background_gradient(subset=["Similitud"], cmap="Greens"),
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()