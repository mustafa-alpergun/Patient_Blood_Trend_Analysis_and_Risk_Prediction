import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Blood Values Early Warning System (Kan Değerleri Erken Uyarı Sistemi)", layout="wide")
st.title("Time Series Based Early Warning System (Zaman Serisi Tabanlı Erken Uyarı Sistemi)")

@st.cache_resource
def load_models():
    try:
        iso_forest = joblib.load('iso_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return iso_forest, scaler
    except FileNotFoundError:
        return None, None

iso_forest, scaler = load_models()

kan_parametreleri = ['ERY', 'HK', 'LEUKO', 'HB', 'PLT', 'MCV', 'MCHC', 'MCH', 'RDW']
egitim_sutunlari = kan_parametreleri + [f'{p}_Velocity' for p in kan_parametreleri]

col1, col2, col3 = st.columns(3)
onceki_degerler = {}
guncel_degerler = {}

with col1:
    st.subheader("Previous Test (T-1) (Önceki Test (T-1))")
    for p in kan_parametreleri:
        onceki_degerler[p] = st.number_input(f"Previous (Önceki) {p}", value=0.0, format="%.2f")

with col2:
    st.subheader("Current Test (T0) (Güncel Test (T0))")
    for p in kan_parametreleri:
        guncel_degerler[p] = st.number_input(f"Current (Güncel) {p}", value=0.0, format="%.2f")

with col3:
    st.subheader("Time Factor (Zaman Faktörü)")
    saat_farki = st.number_input("Time between two tests (Hours) (İki test arası süre (Saat))", min_value=0.1, value=24.0, format="%.1f")
    
    if st.button("Perform Anomaly Analysis (Anomali Analizi Yap)", use_container_width=True):
        if iso_forest is None or scaler is None:
            st.warning("Model files (iso_forest_model.pkl, scaler.pkl) not found. Please add your trained models to the directory. (Model dosyaları (iso_forest_model.pkl, scaler.pkl) bulunamadı. Lütfen eğittiğiniz modelleri dizine ekleyin.)")
        else:
            hesaplanan_ozellikler = {}
            for p in kan_parametreleri:
                hesaplanan_ozellikler[p] = guncel_degerler[p]
                delta = guncel_degerler[p] - onceki_degerler[p]
                hesaplanan_ozellikler[f'{p}_Velocity'] = delta / saat_farki
            
            df_input = pd.DataFrame([hesaplanan_ozellikler])
            df_input = df_input[egitim_sutunlari] 
            
            X_scaled = scaler.transform(df_input)
            anomali_tahmini = iso_forest.predict(X_scaled)[0]
            
            st.markdown("---")
            if anomali_tahmini == -1:
                st.error("🚨 CRITICAL ANOMALY: The rate of change in the patient's blood values shows a dangerous trend! Urgent intervention may be required. (🚨 KRİTİK ANOMALİ: Hastanın kan değerlerindeki değişim hızı tehlikeli bir trend gösteriyor! Acil müdahale gerekebilir.)")
            else:
                st.success("✅ NORMAL COURSE: The change in blood values is within expected limits. (✅ NORMAL SEYİR: Kan değerlerindeki değişim beklenen sınırlar içerisinde.)")

st.markdown("---")
st.subheader("Batch Analysis / File Upload (Toplu Analiz / Dosya Yükleme)")
yuklenen_dosya = st.file_uploader("Upload a CSV or Excel file containing patient data (Hasta verilerini içeren CSV veya Excel dosyası yükleyin)", type=['csv', 'xlsx'])

if yuklenen_dosya is not None and iso_forest is not None and scaler is not None:
    if yuklenen_dosya.name.endswith('.csv'):
        df = pd.read_csv(yuklenen_dosya)
    else:
        df = pd.read_excel(yuklenen_dosya)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['PatientNum', 'Timestamp'])

    for col in kan_parametreleri:
        df[f'{col}_Delta'] = df.groupby('PatientNum')[col].diff().fillna(0)

    df['Saat_Farki'] = df.groupby('PatientNum')['Timestamp'].diff().dt.total_seconds() / 3600
    df['Saat_Farki'] = df['Saat_Farki'].fillna(0).replace(0, 1) 

    for col in kan_parametreleri:
        df[f'{col}_Velocity'] = df[f'{col}_Delta'] / df['Saat_Farki']

    for col in egitim_sutunlari:
        if col not in df.columns:
            df[col] = 0.0

    X_anomaly = df[egitim_sutunlari].fillna(0)
    X_anomaly_scaled = scaler.transform(X_anomaly)
    
    df['Anomaly_Score'] = iso_forest.predict(X_anomaly_scaled)
    df['Risk_Durumu'] = df['Anomaly_Score'].apply(lambda x: 'Critical Anomaly (Kritik Anomali)' if x == -1 else 'Normal Course (Normal Seyir)')

    st.write("### All Patient Data Results (Tüm Hasta Verisi Sonuçları)")
    st.dataframe(df[['PatientNum', 'Timestamp'] + kan_parametreleri + ['Risk_Durumu']])
    
    riskli_hastalar = df[df['Risk_Durumu'] == 'Critical Anomaly (Kritik Anomali)']
    if not riskli_hastalar.empty:
        st.write("### 🚨 Problematic Patients (Problemli Hastalar)")
        st.dataframe(riskli_hastalar[['PatientNum', 'Timestamp'] + kan_parametreleri + ['Anomaly_Score', 'Risk_Durumu']])