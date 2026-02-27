import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 1. Loading and Preparing Data (1. Veriyi Yükleme ve Hazırlama)
df = pd.read_excel(r"C:\Users\muham\OneDrive\Desktop\makine_ogrenmesi_linkedin_projeleri\KL_dataset.xlsx")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 2. Feature Engineering - Time Series and Trend Analysis (2. Feature Engineering (Zaman Serisi ve Trend Analizi))
# Grouping patients and sorting by time to calculate the change (delta) of each blood value compared to the previous test. (Hastaları gruplayıp zamana göre sıralayarak her bir kan değerinin bir önceki teste göre değişimini (delta) hesaplıyoruz.)
df = df.sort_values(by=['PatientNum', 'Timestamp'])
kan_parametreleri = ['ERY', 'HK', 'LEUKO', 'HB', 'PLT', 'MCV', 'MCHC', 'MCH', 'RDW']

for col in kan_parametreleri:
    # Absolute change compared to the previous test (Bir önceki teste göre mutlak değişim)
    df[f'{col}_Delta'] = df.groupby('PatientNum')[col].diff().fillna(0)
    # Percentage change compared to the previous test (Bir önceki teste göre yüzdelik değişim)
    df[f'{col}_Pct_Change'] = df.groupby('PatientNum')[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

# Velocity - change normalized by the hour difference between consecutive tests (Değişim hızı (Velocity) - ardışık testler arasındaki saat farkına göre normalize edilmiş değişim)
df['Saat_Farki'] = df.groupby('PatientNum')['Timestamp'].diff().dt.total_seconds() / 3600
df['Saat_Farki'] = df['Saat_Farki'].fillna(0).replace(0, 1) # To prevent division by zero (Sıfıra bölmeyi engellemek için)

for col in kan_parametreleri:
    df[f'{col}_Velocity'] = df[f'{col}_Delta'] / df['Saat_Farki']

# 3. Unsupervised Learning: Multivariate Anomaly Detection (Isolation Forest) (3. Unsupervised Öğrenme: Çok Değişkenli Anomali Tespiti (Isolation Forest))
# Finds abnormal patterns (hidden risks) shown by the parameters TOGETHER, even if they fit the reference ranges. (Referans aralıklarına uysa bile, parametrelerin BİRLİKTE gösterdiği anormal örüntüleri (gizli riskleri) bulur.)
features_for_anomaly = kan_parametreleri + [f'{col}_Velocity' for col in kan_parametreleri]
X_anomaly = df[features_for_anomaly].fillna(0)

scaler = StandardScaler()
X_anomaly_scaled = scaler.fit_transform(X_anomaly)

iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['Anomaly_Score'] = iso_forest.fit_predict(X_anomaly_scaled)
df['Risk_Durumu'] = df['Anomaly_Score'].apply(lambda x: 'Critical Anomaly (Emergency) (Kritik Anomali (Acil))' if x == -1 else 'Normal Course (Normal Seyir)')

# 4. Supervised Learning: Ward/Triage Prediction (XGBoost) (4. Supervised Öğrenme: Servis/Triage Tahmini (XGBoost))
# We predict which ward (WardNum) the patient should be directed to by looking at their blood profile and change trends. (Hastanın kan profiline ve değişim trendlerine bakarak hangi servise (WardNum) yönlendirilmesi gerektiğini tahmin ederiz.)
# WardNum acts as a "Triage/Urgency" or "Disease Type" label here. (WardNum burada bir "Triage/Aciliyet" veya "Hastalık Tipi" etiketi olarak görev yapar.)
X = df[features_for_anomaly]
y = df['WardNum']

# If the number of classes is too large, let's target only the 5 most common wards (Sampling) (Sınıf sayısı çok fazlaysa sadece en sık rastlanan 5 servisi hedef alalım (Örnekleme))
top_wards = y.value_counts().nlargest(5).index
df_filtered = df[df['WardNum'].isin(top_wards)]
X_filtered = df_filtered[features_for_anomaly]
y_filtered = df_filtered['WardNum']

# Label Encoding (XGBoost requires labels starting from 0) (Label Encoding (XGBoost 0'dan başlayan etiketler ister))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y_filtered)

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

xgb_model = XGBClassifier(
    n_estimators=200, 
    max_depth=6, 
    learning_rate=0.05, 
    objective='multi:softprob', 
    random_state=42,
    tree_method='hist'
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# 5. Reporting Results (5. Sonuçları Raporlama)
print("--- XGBoost Ward/Triage Prediction Performance (XGBoost Servis/Triage Tahmin Performansı) ---")
print(classification_report(y_test, y_pred, target_names=[str(w) for w in le.classes_]))

# Fetching the 5 most at-risk patients (En riskli 5 hastayı getirme)
print("\n--- Top 5 Patients with Most At-Risk (Abnormal) Blood Change Detected by Isolation Forest (Isolation Forest ile Tespit Edilen En Riskli (Anormal) Kan Değişimi Olan 5 Hasta) ---")
riskli_hastalar = df[df['Risk_Durumu'] == 'Critical Anomaly (Emergency) (Kritik Anomali (Acil))'].sort_values('Saat_Farki').head(5)
print(riskli_hastalar[['PatientNum', 'Timestamp', 'HB', 'HB_Delta', 'LEUKO', 'LEUKO_Velocity', 'Risk_Durumu']])