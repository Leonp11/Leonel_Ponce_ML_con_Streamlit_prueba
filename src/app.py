#-----------------------------------------
#Los IMPORTS
#-----------------------------------------
import streamlit as st         
import pandas as pd 
import matplotlib.pyplot as plt            
import joblib                   
from pathlib import Path        

#-----------------------------------------
#Ruta del modelo
#-----------------------------------------
#Ruta del modelo
BASE_DIR = Path(__file__).resolve().parent.parent  # sube un nivel desde src/
MODEL_PATH = BASE_DIR / "models" / "modelo_poliza.pkl"

#Cargamos el modelo
model = joblib.load(MODEL_PATH)


# -----------------------------------------
# Nombramos la app para streamlit
# -----------------------------------------
st.title("Predicción de precio de póliza de seguros")

st.markdown("""
Ingrese sus datos para predecir el precio estimado de su póliza.
""")

# -----------------------------------------
# Inputs del usuario
# -----------------------------------------
age = st.slider(
    label="¿Cuál es tu edad?",
    min_value=12,
    max_value=99,
    value=30,
    help="Desliza y selecciona tu edad"
)

if age <= 12:
    st.info("🧒 Eres un niño aún")
elif age <= 25:
    st.info("👨‍🦱 Eres joven, la vida al descontrol")
elif age <= 40:
    st.info("🧔 comienzan las crisis existenciales")
elif age <= 56:
    st.info("🧔 Ya eres mayor")
elif age <= 70:
    st.info("👴 Ya eres mayor")
else:
    st.warning("💀 La tierra te reclama")

bmi = st.slider(
    label="¿Cuál es tu indice de masa corporal? (BMI)",
    min_value=10.0,
    max_value=50.0,
    value=25.0,
    step=0.1,
    help="Tu indice de masa corporal (IMC) se calcula así: tu peso / tu altura² (kg/m²)"
)

if bmi > 30 and bmi <= 35:
    st.info("🐷 Habrá que cuidar la alimentación")
elif bmi > 35 and bmi <= 40:
    st.warning("💪 Es necesario comenzar el gimnasio")
elif bmi > 40:
    st.error("🔥 Es necesario tomar medidas urgentes")

smoker_num = st.radio(
    label="¿Eres fumador?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Sí",
    help="Selecciona sí o no"
)

if smoker_num == 1:
    st.warning("⚠️ El Fumar incrementa significativamente el precio de la póliza de seguros y amarillenta los dientes")
else:
    st.success("👍 El no fumar reduce riesgos de salud y puede abaratar el costo final de la póliza de seguro")


# Variable dummy según fumador
smoker_yes = 1 if smoker_num == 1 else 0

# -----------------------------------------
# Preparar los datos para el modelo
# -----------------------------------------
X_input = pd.DataFrame([[age, bmi, smoker_num, smoker_yes]], columns=model.feature_names_in_)

# -----------------------------------------
# Botón para "Simular""
# -----------------------------------------
if st.button("Simular ahora"):
    # Predicción del modelo
    pred = model.predict(X_input)
    precio_modelo = pred[0]

    tasa_conversion = 0.0335  # 1 unidad del modelo ≈ 0.0335 euros
    precio_eur = precio_modelo * tasa_conversion
    

    st.write(f"El precio de su póliza de seguro es de: {precio_eur:.2f} € 💶")

    df_compare = pd.DataFrame([
        [age, bmi, 0, 0],
        [age, bmi, 1, 1] 
    ], columns=model.feature_names_in_)

    preds_compare = model.predict(df_compare)
    preds_eur = preds_compare * tasa_conversion  # <-- aquí se define

    # -------------------------------
    # Gráfico de barras
    # -------------------------------

    fig, ax = plt.subplots(figsize=(6,2))

    fig.patch.set_facecolor("#d1d1d1")
    ax.set_facecolor('#d1d1d1')

    ax.barh(["No fumador, gente de bien", "Fumador maloliente"], preds_eur, color=["#1adcb2", "red"],  height=0.2, alpha=0.6)
    ax.set_xlabel("Precio (€)", fontsize=12, fontweight="bold")
    ax.set_title("Simulación de su póliza si fumas o no", fontsize=12, fontweight="bold")


    for i, v in enumerate(preds_eur):
        ax.text(v + 50, i, f"{v:,.2f} €", va='center')

    plt.tight_layout()
    st.pyplot(fig)
