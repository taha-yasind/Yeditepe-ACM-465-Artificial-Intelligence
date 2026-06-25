# ============================================================
# KREDİ KARTI DEFAULT TAHMİNİ —
# ============================================================
# Amaç   : Müşterilerin bir sonraki ay kredi kartı borcunu
#           ödeyip ödeyemeyeceğini tahmin etmek.
# Dosya  : decision_tree.py → Karar Ağacı Modeli
# Veri   : UCI Credit Card Default Dataset (30.000 müşteri)
# ============================================================

# sayısal hesaplamalar için numpy'ı içe aktarıyorum
import numpy as np
# tablo işlemleri için pandas'ı içe aktarıyorum
import pandas as pd
# grafik çizmek için matplotlib'i içe aktarıyorum
import matplotlib.pyplot as plt
# daha şık grafikler için seaborn'u içe aktarıyorum
import seaborn as sns
# veriyi eğitim ve test olarak bölmek için kullanıyorum
from sklearn.model_selection import train_test_split
# karar ağacı modeli ve görselleştirme aracını içe aktarıyorum
from sklearn.tree import DecisionTreeClassifier, plot_tree
# modeli değerlendirmek için kullandığım metrik araçları
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)

# -------------------------------------------------------
# BÖLÜM 1 — VERİYİ OKUMA VE HAZIRLAMA
# -------------------------------------------------------

# veri setini Excel dosyasından okuyorum
df = pd.read_excel('../creditCard_UCI.xlsx')

# ID sütunu sadece müşteri numarası, modele hiçbir katkısı yok, bu yüzden çıkarıyorum
df = df.drop(columns=['ID'])

# X → modele verilecek giriş değişkenleri (özellikler)
# Y → modelin tahmin etmesini istediğim hedef değişken (ödedi mi / ödemedi mi)
X = df.drop(columns=['default.payment.next.month'])
Y = df['default.payment.next.month']

print("=" * 50)
print("VERİ HAZIRLANDL:")
print(f"  X boyutu: {X.shape}")
print(f"  Y boyutu: {Y.shape}")

# -------------------------------------------------------
# BÖLÜM 2 — TRAIN - TEST SPLIT
# -------------------------------------------------------

# veriyi %80 eğitim %20 test olarak ikiye bölüyorum
# stratify=Y → ödeyenler ve ödemeyenlerin oranını her iki sette de koruyorum
# random_state=42 → her çalıştırmada aynı bölünmeyi elde ediyorum
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("\nTRAIN - TEST SPLIT:")
print(f"  Eğitim seti: {X_train.shape[0]:,} müşteri")
print(f"  Test seti  : {X_test.shape[0]:,} müşteri")

# NOT: StandardScaler uygulanmıyor.
# Decision Tree ağaç tabanlı bir modeldir.
# Ağaç modelleri veriyi eşitsizlik noktalarına göre böler,
# değişkenlerin ölçeği kararı etkilemez.

# -------------------------------------------------------
# BÖLÜM 3 — DECISION TREE MODELİ
# -------------------------------------------------------

# karar ağacı modelimi oluşturuyorum ve parametrelerini belirliyorum
# max_depth=5 → ağacın en fazla 5 kat derine inmesine izin veriyorum (aşırı öğrenmeyi önlemek için)
# class_weight='balanced' → veri setindeki sınıf dengesizliğini otomatik olarak düzeltiyor
# random_state=42 → her çalıştırmada aynı sonucu alıyorum
dt_model = DecisionTreeClassifier(
    max_depth=5,
    class_weight='balanced',
    random_state=42
)

# modeli eğitim verisiyle eğitiyorum → model burada kalıpları öğreniyor
dt_model.fit(X_train, Y_train)

print("\n" + "=" * 50)
print("DECISION TREE MODELİ EĞİTİLDİ")

# -------------------------------------------------------
# BÖLÜM 4 — TAHMİN VE DEĞERLENDİRME
# -------------------------------------------------------

# test verisi üzerinde tahmin yapıyorum (model bu veriyi daha önce hiç görmedi)
Y_pred_dt = dt_model.predict(X_test)
# her müşterinin "ödemeyecek" olasılığını alıyorum (sütun 1 = ödemeyecek sınıfı)
Y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

# accuracy → modelin kaç tanesini doğru tahmin ettiğinin oranı
dt_accuracy = accuracy_score(Y_test, Y_pred_dt)
# roc_auc → modelin ödeyeni ve ödemeyeni ne kadar iyi ayırt ettiğini gösteren skor (0.5-1 arası, 1 mükemmel)
dt_roc_auc  = roc_auc_score(Y_test, Y_prob_dt)

print("\n" + "=" * 50)
print("DECISION TREE — SONUÇLAR:")
print(f"  Accuracy (Doğruluk) : %{dt_accuracy * 100:.2f}")
print(f"  ROC-AUC Skoru       : {dt_roc_auc:.4f}")

# her sınıf için precision, recall ve f1 skorlarını detaylı gösteriyorum
print("\nDETAYLI SINIFLANDIRMA RAPORU:")
print(classification_report(Y_test, Y_pred_dt,
                             target_names=['Ödedi (0)', 'Ödemedi (1)']))

# confusion matrix → modelin hangi tahminlerde yanıldığını gösteren tablo
cm_dt = confusion_matrix(Y_test, Y_pred_dt)
print("KARMAŞIKLIK MATRİSİ (Confusion Matrix):")
print(cm_dt)
# matrisin her hücresinin ne anlama geldiğini açıklıyorum
print(f"  Doğru tahmin (Ödedi)   : {cm_dt[0][0]}")
print(f"  Yanlış alarm           : {cm_dt[0][1]}")
print(f"  Kaçırılan default      : {cm_dt[1][0]}")
print(f"  Doğru tahmin (Ödemedi): {cm_dt[1][1]}")

# -------------------------------------------------------
# BÖLÜM 5 — GRAFİKLER
# -------------------------------------------------------

# confusion matrix'i görsel olarak çiziyorum
plt.figure(figsize=(6, 5))
# heatmap ile matristeki sayıları renkli kutular içinde gösteriyorum
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ödedi (0)', 'Ödemedi (1)'],
            yticklabels=['Ödedi (0)', 'Ödemedi (1)'])
plt.title('Decision Tree — Confusion Matrix')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen')
plt.tight_layout()
plt.savefig('dt_confusion_matrix.png', dpi=150)
plt.show()
print("→ Grafik kaydedildi: dt_confusion_matrix.png")

# karar ağacının yapısını görselleştiriyorum (ilk 3 seviyeyi gösteriyorum, tamamı çok karmaşık olur)
plt.figure(figsize=(20, 8))
plot_tree(dt_model,
          max_depth=3,
          feature_names=X.columns.tolist(),
          class_names=['Ödedi', 'Ödemedi'],
          filled=True,       # düğümleri renkle dolduruyor
          rounded=True,      # köşeleri yuvarlak yapıyor
          fontsize=10)
plt.title('Decision Tree Görselleştirmesi (İlk 3 Seviye)')
plt.tight_layout()
plt.savefig('dt_agac_gorseli.png', dpi=150)
plt.show()
print("→ Grafik kaydedildi: dt_agac_gorseli.png")

# modelin en çok hangi değişkenlere baktığını hesaplıyorum ve ilk 10'unu seçiyorum
dt_importance = pd.DataFrame({
    'Değişken': X.columns,
    'Önem': dt_model.feature_importances_
}).sort_values('Önem', ascending=False).head(10)

print("\nEN ÖNEMLİ 10 DEĞİŞKEN:")
print(dt_importance.to_string(index=False))

# en önemli değişkenleri yatay bar grafiğiyle görselleştiriyorum
plt.figure(figsize=(8, 5))
# [::-1] → listeyi ters çeviriyorum, en önemli değişken en üstte görünsün diye
plt.barh(dt_importance['Değişken'][::-1],
         dt_importance['Önem'][::-1],
         color='steelblue')
plt.title('Decision Tree — En Önemli Değişkenler')
plt.xlabel('Önem Skoru')
plt.tight_layout()
plt.savefig('dt_feature_importance.png', dpi=150)
plt.show()
print("→ Grafik kaydedildi: dt_feature_importance.png")

# -------------------------------------------------------
# ÖZET
# -------------------------------------------------------

print("\n" + "=" * 50)
print("DECISION TREE TAMAMLANDI.")
print(f"  Accuracy : %{dt_accuracy * 100:.2f}")
print(f"  ROC-AUC  : {dt_roc_auc:.4f}")
print("=" * 50)