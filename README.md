Hello
In this project, I developed an end-to-end medical early warning system that detects hidden risks and predicts triage wards from patient blood value trends using a dual machine learning approach.
Project Overview:
🔹 Dual Model Architecture: I designed a system combining Isolation Forest for unsupervised multivariate anomaly detection and XGBoost for supervised triage/ward prediction.
🔹 Objective: To automate the detection of critical health risks by analyzing the velocity of changes in blood parameters (like HB, LEUKO, PLT) over time.
🔹 Technical Pipeline:
Feature Engineering: Calculated absolute changes (deltas) and time-normalized change rates (velocities) between consecutive blood tests.
Model Integration: Used Isolation Forest to find hidden risky patterns across multiple parameters simultaneously, and XGBoost to predict the appropriate hospital ward based on the patient's blood profile.
Web Deployment: Developed a functional web application using Streamlit, enabling both manual data entry and batch analysis via CSV/Excel uploads for real-time risk assessment.
🔹 Evaluation: Validated the XGBoost model using Classification Reports to ensure accurate triage routing.
Tech Stack: Python, Pandas, Scikit-Learn, XGBoost, Streamlit.
If you'd like, you can try it out by examining the website I prepared for testing purposes.
Feel free to review the code and share your feedback!
Author: Mustafa Alpergün

Merhaba
Bu projede, hastaların kan değeri trendlerini analiz ederek gizli riskleri tespit eden ve triyaj servisi tahmini yapan, makine öğrenmesi tabanlı uçtan uca bir erken uyarı sistemi geliştirdim.
Proje Detayları:
🔹 İkili Model Mimarisi: Gözetimsiz çok değişkenli anomali tespiti için Isolation Forest ve gözetimli triyaj/servis tahmini için XGBoost algoritmalarını birleştiren bir yapı tasarladım.
🔹 Amaç: Kan parametrelerindeki (HB, LEUKO, PLT vb.) zamana bağlı değişim hızlarını analiz ederek kritik sağlık risklerini otomatik olarak tespit etmek.
🔹 Teknik Süreç ve Dağıtım:
Özellik Çıkarımı (Feature Engineering): Ardışık kan testleri arasındaki mutlak değişimleri (delta) ve zamana göre normalize edilmiş değişim hızlarını (velocity) hesapladım.
Model Entegrasyonu: Referans aralıklarına uysa bile parametrelerin birlikte gösterdiği riskli örüntüleri bulmak için Isolation Forest, hastanın profiline göre en uygun servisi tahmin etmek için XGBoost kullandım.
Web Entegrasyonu: Eğitilen modelleri Streamlit ile web uygulamasına dönüştürdüm; hem manuel giriş hem de toplu dosya (CSV/Excel) yükleme ile anlık risk analizi sağladım.
🔹 Performans: XGBoost modelinin başarısı detaylı sınıflandırma raporları (Classification Report) ile analiz edildi.
Kullanılan Teknolojiler: Python, Pandas, Scikit-Learn, XGBoost, Streamlit.
Dilerseniz, test amacıyla hazırladığım web sitesini inceleyerek deneyebilirsiniz.
Kodları incelemek ve geliştirme önerilerinizi paylaşmak isterseniz geri bildirimleriniz benim için çok değerli!
Yazar: Mustafa Alpergün
