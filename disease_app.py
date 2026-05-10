import streamlit as st
import pandas as pd
import pickle
import time
 

# 1. CHARGER LE FICHIER UNIQUE

with open('plant_disease_free_model.pkl', 'rb') as file:
    model = pickle.load(file)
 
classifier = model['modele']
Ohe        = model['ohe']
sc         = model['scaler']
 
# 2. CONFIGURATION + CSS

st.set_page_config(
    page_title="Plant Disease Predictor",
    page_icon="🌿",
    layout="centered"
)
 
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 50%, #e0f2f1 100%);
    }
    h1 { color: #1B5E20 !important; text-align: center; font-size: 2.2rem !important; }
    h3 { color: #2E7D32 !important; }
    .stButton > button {
        background: linear-gradient(90deg, #2E7D32, #388E3C) !important;
        color: white !important;
        border-radius: 14px !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        padding: 12px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(46,125,50,0.3) !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1B5E20, #2E7D32) !important;
        box-shadow: 0 6px 18px rgba(46,125,50,0.5) !important;
        transform: translateY(-2px) !important;
    }
    .result-blight {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        border-left: 6px solid #E53935;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 3px 10px rgba(229,57,53,0.15);
    }
    .result-mildew {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border-left: 6px solid #1E88E5;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 3px 10px rgba(30,136,229,0.15);
    }
    .result-rust {
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
        border-left: 6px solid #FB8C00;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 3px 10px rgba(251,140,0,0.15);
    }
    .footer {
        text-align: center;
        color: #777;
        font-size: 12px;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #C8E6C9;
    }
</style>
""", unsafe_allow_html=True)
 

# 3. HEADER

st.markdown("<h1>🌿 Plant Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#4CAF50; font-size:16px; font-weight:500;'>
    🤖 Intelligence Artificielle · Détection de maladies des plantes médicinales
</p>
""", unsafe_allow_html=True)
st.divider()
 

# 4. PHOTOS DES 3 MALADIES

st.markdown("### 🌱 Les 3 maladies que je détecte")
 
col_a, col_b, col_c = st.columns(3)
 
with col_a:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/Tomato_-_late_blight.jpg/320px-Tomato_-_late_blight.jpg",
        caption="🔴 Blight — Mildiou",
        use_container_width=True
    )
    st.markdown("""
    <div style='background:#FFEBEE;border-radius:8px;padding:8px;text-align:center;font-size:12px;color:#C62828'>
        Attaque les feuilles et tiges
    </div>""", unsafe_allow_html=True)
 
with col_b:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Powdery_mildew_on_a_courgette_leaf.jpg/320px-Powdery_mildew_on_a_courgette_leaf.jpg",
        caption="🔵 Mildew — Oïdium",
        use_container_width=True
    )
    st.markdown("""
    <div style='background:#E3F2FD;border-radius:8px;padding:8px;text-align:center;font-size:12px;color:#1565C0'>
        Poudre blanche sur les feuilles
    </div>""", unsafe_allow_html=True)
 
with col_c:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Rust_on_wheat.jpg/320px-Rust_on_wheat.jpg",
        caption="🟠 Rust — Rouille",
        use_container_width=True
    )
    st.markdown("""
    <div style='background:#FFF3E0;border-radius:8px;padding:8px;text-align:center;font-size:12px;color:#E65100'>
        Taches orangées sur les feuilles
    </div>""", unsafe_allow_html=True)
 
st.divider()
 

# 5. FORMULAIRE DE SAISIE

st.markdown("### 🔬 Caractéristiques de la plante")
st.caption("Remplis les informations ci-dessous puis clique sur Prédire")
 
col1, col2 = st.columns(2)
 
with col1:
    leaf_length   = st.slider("🍃 Longueur des feuilles (cm)", 3.0, 18.0, 10.0)
    leaf_width    = st.slider("🍃 Largeur des feuilles (cm)",  0.5,  9.0,  5.0)
    stem_diameter = st.slider("🌱 Diamètre de la tige (cm)",  0.1,  1.8,  1.0)
 
with col2:
    soil_type = st.selectbox("🪨 Type de sol",         ["clay", "loamy", "sandy"],
                             format_func=lambda x: {"clay":"🟤 Argileux (clay)",
                                                     "loamy":"🟡 Limoneux (loamy)",
                                                     "sandy":"⚪ Sableux (sandy)"}[x])
    weather   = st.selectbox("⛅ Météo",               ["sunny", "rainy", "cloudy"],
                             format_func=lambda x: {"sunny":"☀️ Ensoleillé",
                                                     "rainy":"🌧️ Pluvieux",
                                                     "cloudy":"☁️ Nuageux"}[x])
    pesticide = st.selectbox("🧪 Pesticide utilisé ?", ["no", "yes"],
                             format_func=lambda x: {"no":"❌ Non", "yes":"✅ Oui"}[x])
 
st.divider()
 

# 6. PRÉDICTION

if st.button("🔍 Prédire la maladie", use_container_width=True):
 
    with st.spinner("🤖 Analyse de la plante en cours..."):
        time.sleep(1)
 
        # Étape A : Encodage manuel de pesticide
        pesti_encoded = 1 if pesticide == "yes" else 0
 
        # Étape B : OHE sur soil_type et weather uniquement
        cat_data  = pd.DataFrame([{'soil_type': soil_type, 'weather': weather}])
        cat_cols  = ['soil_type', 'weather']
        encoded   = Ohe.transform(cat_data[cat_cols])
        col_names = Ohe.get_feature_names_out(cat_cols)
        df_encoded = pd.DataFrame(encoded.astype(int), columns=col_names)
 
        # Étape C : Construire new_plant dans l'ordre EXACT du notebook
        new_plant = pd.DataFrame([{
            'leaf_length':     leaf_length,
            'leaf_width':      leaf_width,
            'stem_diameter':   stem_diameter,
            'pesticide':       pesti_encoded,
            'soil_type_clay':  df_encoded['soil_type_clay'].values[0],
            'soil_type_loamy': df_encoded['soil_type_loamy'].values[0],
            'soil_type_sandy': df_encoded['soil_type_sandy'].values[0],
            'weather_cloudy':  df_encoded['weather_cloudy'].values[0],
            'weather_rainy':   df_encoded['weather_rainy'].values[0],
            'weather_sunny':   df_encoded['weather_sunny'].values[0],
        }])
 
        # Étape D : StandardScaler
        new_plant_scaled = sc.transform(new_plant)
 
        # Étape E : Prédiction KNN
        pred_encoded = classifier.predict(new_plant_scaled)
        classes_dict = {0: 'blight', 1: 'mildew', 2: 'rust'}
        pred_label   = classes_dict[int(pred_encoded[0])]
        probas       = classifier.predict_proba(new_plant_scaled)[0]
        classes      = ['blight', 'mildew', 'rust']
        best_prob    = max(probas)
 
    # Résultat coloré avec description
    descriptions = {
        'blight': ('🔴', 'BLIGHT — Mildiou',
                   'Maladie fongique qui attaque les feuilles, tiges et fruits. '
                   'Se développe par temps humide et frais.'),
        'mildew': ('🔵', 'MILDEW — Oïdium',
                   'Champignon formant une poudre blanche sur les feuilles. '
                   'Favorisé par la chaleur et l\'humidité modérée.'),
        'rust':   ('🟠', 'RUST — Rouille',
                   'Champignon parasite créant des taches orangées en relief. '
                   'Se propage par le vent et les éclaboussures d\'eau.')
    }
 
    icon, titre, desc = descriptions[pred_label]
 
    st.markdown(f"""
    <div class="result-{pred_label}">
        <h3 style="margin:0 0 8px 0">{icon} Maladie détectée : {titre}</h3>
        <p style="margin:0; color:#444; font-size:14px">{desc}</p>
    </div>
    """, unsafe_allow_html=True)
 
    # Probabilités
    st.markdown("### 📊 Niveau de confiance par maladie")
    icons_cls = {'blight': '🔴', 'mildew': '🔵', 'rust': '🟠'}
    for cls, prob in zip(classes, probas):
        st.write(f"{icons_cls[cls]} **{cls.upper()}**")
        st.progress(float(prob), text=f"{prob*100:.1f}%")
 
    # Ballons + message confiance
    if best_prob >= 0.5:
        st.balloons()
        st.success(f"🎯 Bonne confiance du modèle : **{best_prob*100:.1f}%**")
    else:
        st.warning(f"⚠️ Confiance faible ({best_prob*100:.1f}%) — plusieurs maladies sont proches.")
 

# 7. FOOTER

st.divider()
st.markdown("""
<div class="footer">
    🌿 <strong>Plant Disease Predictor</strong> · Algorithme K-Nearest Neighbors (KNN)<br>
    Projet académique de Machine Learning · Dataset : 540 plantes médicinales
</div>
""", unsafe_allow_html=True)