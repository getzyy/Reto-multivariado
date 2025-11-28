import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Interlub AI Suite",
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
    h3 { color: #004e92; }
    /* Ajuste para que los tooltips se vean bien */
    div[data-testid="stMetricLabel"] > div:nth-child(2) {
        color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA DE DATOS (Cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    try:
        df_base = pd.read_csv("final_df_final.csv")
        df_anderol = pd.read_csv("competencia/Anderol.csv")
        df_solgelsal = pd.read_csv("competencia/Colgelsa - Hoja 1.csv")
        df_starplex = pd.read_csv("competencia/Starplex.csv")
    except FileNotFoundError:
        st.error("Error crítico: No se encuentran los archivos CSV. Verifica las rutas.")
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

    for col in columnas_modelo:
        if col not in df_base.columns: df_base[col] = np.nan
        if col not in df_anderol.columns: df_anderol[col] = np.nan

    df_base = df_base[columnas_modelo].copy()
    df_anderol = df_anderol[columnas_modelo].copy()
    df_solgelsal = df_solgelsal[columnas_modelo].copy()
    df_starplex = df_starplex[columnas_modelo].copy()

    df_base["Empresa"] = "Interlub"
    df_anderol["Empresa"] = "Anderol"
    df_solgelsal["Empresa"] = "Solgelsal"
    df_starplex["Empresa"] = "Starplex"

    df_total = pd.concat([df_base, df_anderol, df_solgelsal, df_starplex], ignore_index=True)
    df_total = df_total.reset_index(drop=True)

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
# 2. PROCESAMIENTO MATRICES Y MODELOS
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

    X_num = df_total[feature_cols].copy()
    for col in feature_cols:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
    
    means = X_num.mean()
    X_num = X_num.fillna(means)
    
    scaler_num = StandardScaler()
    X_num_scaled = scaler_num.fit_transform(X_num)
    sim_num = cosine_similarity(X_num_scaled, X_num_scaled)

    count_vec = CountVectorizer(stop_words=None)
    count_matrix = count_vec.fit_transform(df_total["soup"])
    sim_meta = cosine_similarity(count_matrix, count_matrix)

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
        "sim_meta_norm": sim_meta_norm
    }
    
    return models, X_num

# -----------------------------------------------------------------------------
# 3. LÓGICA DE REGRESIÓN MATRICIAL (OLS)
# -----------------------------------------------------------------------------
def calculate_ols(df, x_vars, y_var):
    data = df[x_vars + [y_var]].dropna()
    
    if len(data) < len(x_vars) + 2:
        return None, "Insuficientes datos para regresión"

    Y = data[y_var].values
    X = data[x_vars].values
    n = len(Y)
    k = len(x_vars)
    
    X_mat = np.column_stack([np.ones(n), X])
    
    try:
        XtX = X_mat.T @ X_mat
        XtX_inv = np.linalg.inv(XtX)
        beta_hat = XtX_inv @ X_mat.T @ Y
    except np.linalg.LinAlgError:
        return None, "Matriz singular (variables correlacionadas). No se puede invertir."

    Y_hat = X_mat @ beta_hat
    epsilon = Y - Y_hat 

    p = k + 1 
    SSE = epsilon.T @ epsilon
    sigma_sq = SSE / (n - p)
    sigma = np.sqrt(sigma_sq)

    cov_matrix = sigma_sq * XtX_inv
    std_errors = np.sqrt(np.diag(cov_matrix))

    Syy = np.sum((Y - np.mean(Y))**2)
    SSR = Syy - SSE
    R2 = 1 - (SSE / Syy)
    
    F_stat = (SSR / k) / (SSE / (n - p)) if k > 0 else 0
    p_value_F = 1 - stats.f.cdf(F_stat, k, n - p)

    aic = n * np.log(SSE/n) + 2*p
    bic = n * np.log(SSE/n) + p * np.log(n)

    t_stats = beta_hat / std_errors
    p_values_t = [2 * (1 - stats.t.cdf(abs(t), n - p)) for t in t_stats]

    results = {
        "beta": beta_hat,
        "std_errors": std_errors,
        "t_stats": t_stats,
        "p_values": p_values_t,
        "R2": R2,
        "MSE": SSE/n,
        "RMSE": np.sqrt(SSE/n),
        "F_stat": F_stat,
        "p_value_F": p_value_F,
        "AIC": aic,
        "BIC": bic,
        "Y_real": Y,
        "Y_pred": Y_hat,
        "Residuos": epsilon,
        "X_data": data[x_vars],
        "n": n
    }
    return results, None

# -----------------------------------------------------------------------------
# 4. LÓGICA DE RECOMENDACIÓN
# -----------------------------------------------------------------------------
def get_recommendations_existing(idx_input, df, models, alpha, filter_competitors=False):
    sim_num = models["sim_num_norm"]
    sim_meta = models["sim_meta_norm"]
    hybrid_row = alpha * sim_num[idx_input] + (1 - alpha) * sim_meta[idx_input]
    sim_scores = sorted(list(enumerate(hybrid_row)), key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx_input]
    indices = [s[0] for s in sim_scores]
    scores = [s[1] for s in sim_scores]
    df_results = df.iloc[indices].copy()
    df_results["Similitud"] = scores
    if filter_competitors:
        df_results = df_results[df_results["Empresa"] == "Interlub"]
    return df_results.head(5)

def get_recommendations_from_attributes(user_input, df_total, models, alpha):
    input_vector = models["means"].copy()
    filled_count = 0
    for col, val in user_input["numeric"].items():
        if val is not None and val != 0: 
            input_vector[col] = val
            filled_count += 1
    if filled_count < 2: return None, "Ingresa al menos 2 valores numéricos."

    input_scaled = models["scaler_num"].transform(input_vector.values.reshape(1, -1))
    sim_num_new = cosine_similarity(input_scaled, models["X_num_scaled"]).flatten()
    sim_num_new_norm = (sim_num_new - sim_num_new.min()) / (sim_num_new.max() - sim_num_new.min() + 1e-9)

    user_soup = user_input.get("soup", "")
    if user_soup.strip() == "":
        sim_meta_new_norm = np.zeros(len(df_total))
    else:
        input_count = models["count_vec"].transform([user_soup])
        sim_meta_new = cosine_similarity(input_count, models["count_matrix"]).flatten()
        sim_meta_new_norm = (sim_meta_new - sim_meta_new.min()) / (sim_meta_new.max() - sim_meta_new.min() + 1e-9)

    hybrid_scores = alpha * sim_num_new_norm + (1 - alpha) * sim_meta_new_norm
    indices = np.argsort(hybrid_scores)[::-1][:5]
    df_results = df_total.iloc[indices].copy()
    df_results["Similitud"] = hybrid_scores[indices]
    return df_results, None

# -----------------------------------------------------------------------------
# 5. VISUALIZACIÓN
# -----------------------------------------------------------------------------
def plot_radar_chart(df_raw, row_data_dict, idx_recomendado, nombre_rec):
    cols_radar = ["Grado NLGI Consistencia", "Punto de Gota, °C", 
                  "Punto de Soldadura Cuatro Bolas, kgf", "Temperatura de Servicio °C, max"]
    
    # Extracción y limpieza forzosa de datos del usuario
    vals_user = []
    for c in cols_radar:
        raw_val = 0
        if isinstance(row_data_dict, pd.Series): 
            raw_val = row_data_dict.get(c, 0)
        else: 
            raw_val = row_data_dict.get(c, 0)
        
        # Convertir a float de manera segura (evita errores con strings)
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            val = 0.0
        vals_user.append(val)
    
    vals_user = np.array(vals_user)
    
    # Extracción y limpieza forzosa de datos recomendados
    raw_rec_vals = df_raw.iloc[idx_recomendado][cols_radar].fillna(0).values
    vals_rec = []
    for v in raw_rec_vals:
        try:
            vals_rec.append(float(v))
        except (ValueError, TypeError):
            vals_rec.append(0.0)
    vals_rec = np.array(vals_rec)

    # Normalización para gráfico
    combined = np.vstack([vals_user, vals_rec])
    max_vals = combined.max(axis=0)
    max_vals[max_vals == 0] = 1  # Evitar división por cero
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_user/max_vals, theta=cols_radar, fill='toself', name='Selección', line_color='blue'))
    fig.add_trace(go.Scatterpolar(r=vals_rec/max_vals, theta=cols_radar, fill='toself', name='Recomendada', line_color='green'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=300, margin=dict(l=40, r=40, t=20, b=20))
    return fig

# -----------------------------------------------------------------------------
# 6. PESTAÑA REGRESIÓN (Con Tooltips agregados)
# -----------------------------------------------------------------------------
def render_regression_tab(df_total, numeric_cols):
    st.markdown("### 📈 Modelado de Regresión Multivariable (OLS)")
    st.caption("Implementación de la teoría matricial para explicar variables de desempeño.")

    col_conf, col_viz = st.columns([1, 3])

    with col_conf:
        st.markdown("#### Configuración del Modelo")
        
        y_target = st.selectbox("Variable Objetivo ($y$)", numeric_cols, index=4, key="reg_y")
        
        x_options = [c for c in numeric_cols if c != y_target]
        x_predictors = st.multiselect(
            "Variables Explicativas ($X$)", 
            x_options, 
            default=x_options[:2] if len(x_options) >=2 else x_options,
            key="reg_x"
        )

        st.info("El modelo calcula: $\\hat{\\beta} = (X^T X)^{-1} X^T y$")
        
        if st.button("Calcular Modelo", key="btn_calc_reg"):
            if not x_predictors:
                st.warning("Selecciona al menos una variable X.")
                return
            
            results, error = calculate_ols(df_total, x_predictors, y_target)
            
            if error:
                st.error(error)
                return
            
            st.session_state['reg_results'] = results
            st.session_state['snapshot_x'] = x_predictors
            st.session_state['snapshot_y'] = y_target

    if 'reg_results' in st.session_state:
        res = st.session_state['reg_results']
        
        current_x = st.session_state.get('reg_x')
        current_y = st.session_state.get('reg_y')
        snapshot_x = st.session_state.get('snapshot_x', [])
        snapshot_y = st.session_state.get('snapshot_y', "")

        if current_x != snapshot_x or current_y != snapshot_y:
             with col_viz: 
                 st.warning("⚠️ La configuración ha cambiado. Haz clic en 'Calcular Modelo' para actualizar los gráficos.")

        with col_viz:
            # --- SECCIÓN DE MÉTRICAS CON TOOLTIPS (HELP) ---
            m1, m2, m3, m4 = st.columns(4)
            
            m1.metric(
                "R² Score", 
                f"{res['R2']:.3f}",
                help="Coeficiente de Determinación: Porcentaje de la varianza de Y explicada por el modelo. (0 a 1, mientras más alto mejor)."
            )
            
            m2.metric(
                "RMSE", 
                f"{res['RMSE']:.2f}",
                help="Raíz del Error Cuadrático Medio: Mide el error promedio en las mismas unidades que la variable Y. (Mientras más bajo mejor)."
            )
            
            m3.metric(
                "F-Statistic", 
                f"{res['F_stat']:.2f}",
                help="Estadístico F: Prueba si el modelo globalmente tiene capacidad predictiva significativa. Un valor alto sugiere que al menos una variable X es relevante."
            )
            
            m4.metric(
                "AIC", 
                f"{res['AIC']:.1f}",
                help="Criterio de Información de Akaike: Mide la calidad del modelo penalizando la complejidad (número de variables). Útil para comparar modelos: el que tenga menor AIC es preferible."
            )
            # ------------------------------------------------

            st.markdown("##### Coeficientes Estimados ($\hat{\\beta}$)")
            coef_df = pd.DataFrame({
                "Variable": ["Intercepto"] + snapshot_x,
                "Coeficiente": res['beta'],
                "Error Std": res['std_errors'],
                "t-value": res['t_stats'],
                "p-value": res['p_values']
            })
            st.dataframe(coef_df.style.format({
                "Coeficiente": "{:.4f}", "Error Std": "{:.4f}", "t-value": "{:.2f}", "p-value": "{:.4f}"
            }), use_container_width=True, hide_index=True)

            tab_g1, tab_g2, tab_g3 = st.tabs(["Dispersión 2D", "Residuales", "Espacio 3D"])
            
            with tab_g1:
                if snapshot_x:
                    var_x_plot = st.selectbox("Eje X para visualizar:", snapshot_x, key="plot_x_sel")
                    
                    fig = px.scatter(
                        res['X_data'], x=var_x_plot, y=res['Y_real'], 
                        labels={'y': snapshot_y}, title=f"Relación: {var_x_plot} vs {snapshot_y}",
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="p_scatter_2d")

            with tab_g2:
                c_r1, c_r2 = st.columns(2)
                fig_hist = px.histogram(x=res['Residuos'], nbins=20, title="Distribución de Residuos ($\epsilon$)", labels={'x': 'Error'})
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                c_r1.plotly_chart(fig_hist, use_container_width=True, key="p_hist_res")
                
                fig_pred = px.scatter(x=res['Y_real'], y=res['Y_pred'], title="Observado vs Predicho", labels={'x':'Real', 'y':'Predicho'})
                fig_pred.add_shape(type="line", line=dict(dash='dash'), x0=res['Y_real'].min(), y0=res['Y_real'].min(), x1=res['Y_real'].max(), y1=res['Y_real'].max())
                c_r2.plotly_chart(fig_pred, use_container_width=True, key="p_pred_real")

            with tab_g3:
                if len(snapshot_x) >= 2:
                    st.markdown("Visualización de los 2 predictores más influyentes y el objetivo.")
                    fig_3d = px.scatter_3d(
                        x=res['X_data'][snapshot_x[0]], 
                        y=res['X_data'][snapshot_x[1]], 
                        z=res['Y_real'],
                        labels={'x': snapshot_x[0], 'y': snapshot_x[1], 'z': snapshot_y},
                        color=res['Y_real'],
                        title="Interacción Multivariable"
                    )
                    st.plotly_chart(fig_3d, use_container_width=True, key="p_3d")
                else:
                    st.info("Se necesitan al menos 2 variables predictoras para el gráfico 3D.")


# -----------------------------------------------------------------------------
# 7. UI PRINCIPAL
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

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏢 Análisis Interno", 
        "⚔️ Benchmarking", 
        "🛠️ Búsqueda por Atributos",
        "📊 Análisis de Regresión" 
    ])

    def render_analysis(mode_internal, unique_key_id):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Grasa de Referencia")
            opciones = df_total["nombre_legible"].tolist()
            seleccion = st.selectbox("Selecciona grasa base:", opciones, key=f"select_{unique_key_id}")
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
                
                # AQUÍ ESTÁ EL CAMBIO CLAVE:
                # Usamos df_numeric_raw para los DATOS SELECCIONADOS TAMBIÉN
                # Esto asegura que enviamos números limpios al gráfico, no strings sucios
                fig = plot_radar_chart(df_numeric_raw, df_numeric_raw.iloc[idx_sel], idx_best, top["nombre_legible"])
                
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{unique_key_id}")
                st.dataframe(results[["Empresa", "nombre_legible", "Similitud", "Aceite Base", "Espesante"]].style.background_gradient(subset=["Similitud"], cmap="Greens"), use_container_width=True)

    with tab1: render_analysis(True, "interno")
    with tab2: render_analysis(False, "competencia")

    with tab3:
        st.markdown("#### 🛠️ Diseña tu grasa ideal")
        with st.form("form_atributos"):
            col_a, col_b, col_c = st.columns(3)
            val_nlgi = col_a.number_input("Grado NLGI", 0.0, 3.0, 0.0, 0.5)
            val_gota = col_b.number_input("Punto de Gota (°C)", 0.0, 500.0, 0.0)
            val_4bolas = col_c.number_input("Carga 4 Bolas (kgf)", 0.0, 1000.0, 0.0)
            col_d, col_e, col_f = st.columns(3)
            val_visc = col_d.number_input("Viscosidad 40°C", 0.0, 1000.0, 0.0)
            val_temp_max = col_e.number_input("Temp Max", 0.0, 500.0, 0.0)
            val_factor = col_f.number_input("Factor Velocidad", 0.0, 1000000.0, 0.0)
            st.markdown("---")
            val_texto = st.text_input("Palabras clave", placeholder="Ej: Litio complejo sintética")
            if st.form_submit_button("🔍 Buscar"):
                user_dict = {
                    "numeric": {
                        "Grado NLGI Consistencia": val_nlgi if val_nlgi>0 else None,
                        "Punto de Gota, °C": val_gota if val_gota>0 else None,
                        "Punto de Soldadura Cuatro Bolas, kgf": val_4bolas if val_4bolas>0 else None,
                        "Viscosidad del Aceite Base a 40°C. cSt": val_visc if val_visc>0 else None,
                        "Temperatura de Servicio °C, max": val_temp_max if val_temp_max>0 else None,
                        "Factor de Velocidad": val_factor if val_factor>0 else None
                    }, "soup": val_texto
                }
                results, error = get_recommendations_from_attributes(user_dict, df_total, models, alpha)
                if results is None: 
                    return
                if error: st.error(error)
                else:
                    top = results.iloc[0]
                    fig = plot_radar_chart(df_numeric_raw, user_dict["numeric"], results.index[0], top["nombre_legible"])
                    st.plotly_chart(fig, use_container_width=True, key="chart_manual")
                    st.dataframe(results[["Empresa", "nombre_legible", "Similitud"]], use_container_width=True)

    with tab4:
        render_regression_tab(df_numeric_raw, models["feature_cols"])

if __name__ == "__main__":
    main()