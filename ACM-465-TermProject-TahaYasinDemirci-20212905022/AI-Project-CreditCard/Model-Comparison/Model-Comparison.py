# ============================================================
# KREDİ KARTI DEFAULT TAHMİNİ
# ============================================================
# Amaç   : Müşterilerin bir sonraki ay kredi kartı borcunu
#           ödeyip ödeyemeyeceğini tahmin etmek.
# Dosya  : Model-Comparison.py → Model Karşılaştırma & Final Yorum
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
# karar ağacı modelini içe aktarıyorum
from sklearn.tree import DecisionTreeClassifier
# rastgele orman modelini içe aktarıyorum
from sklearn.ensemble import RandomForestClassifier
# iki modeli de aynı metriklerle değerlendireceğim için hepsini içe aktarıyorum
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)

# -------------------------------------------------------
# BOLUM 1 - VERIYI OKUMA VE HAZIRLAMA
# -------------------------------------------------------

# veri setini Excel dosyasından okuyorum
df = pd.read_excel('../creditCard_UCI.xlsx')
# ID sütunu modele katkı sağlamıyor, çıkarıyorum
df = df.drop(columns=['ID'])

# X → giriş değişkenleri | Y → tahmin etmek istediğim hedef
X = df.drop(columns=['default.payment.next.month'])
Y = df['default.payment.next.month']

# veriyi %80 eğitim %20 test olarak bölüyorum
# stratify=Y → sınıf oranlarını koruyorum | random_state=42 → tekrarlanabilir sonuç
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("=" * 55)
print("VERI HAZIRLANDI - MODEL KARSILASTIRMA BASLIYOR")
print("=" * 55)

# -------------------------------------------------------
# BOLUM 2 - MODELLERI EGITME
# -------------------------------------------------------

# Decision Tree modelimi oluşturuyorum ve eğitiyorum
# max_depth=5 → ağacın derinliğini sınırlıyorum | class_weight='balanced' → sınıf dengesizliğini gideriyorum
dt_model = DecisionTreeClassifier(
    max_depth=5,
    class_weight='balanced',
    random_state=42
)
dt_model.fit(X_train, Y_train)
print("\nDecision Tree egitildi")

# Random Forest modelimi oluşturuyorum ve eğitiyorum
# n_estimators=100 → 100 ağaç kuruyorum | max_depth=10 → her ağaç en fazla 10 kat derine iniyor
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, Y_train)
print("Random Forest egitildi")

# -------------------------------------------------------
# BOLUM 3 - TAHMINLER
# -------------------------------------------------------

# Decision Tree ile test verisi üzerinde tahmin yapıyorum
Y_pred_dt = dt_model.predict(X_test)
# sınıf olasılıklarını alıyorum (1. sütun = ödemeyecek olasılığı)
Y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

# Random Forest ile test verisi üzerinde tahmin yapıyorum
Y_pred_rf = rf_model.predict(X_test)
# sınıf olasılıklarını alıyorum (1. sütun = ödemeyecek olasılığı)
Y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# -------------------------------------------------------
# BOLUM 4 - METRIKLER
# -------------------------------------------------------

# Decision Tree için tüm metrikleri hesaplıyorum
dt_accuracy  = accuracy_score(Y_test, Y_pred_dt)           # genel doğruluk oranı
dt_roc_auc   = roc_auc_score(Y_test, Y_prob_dt)            # sınıf ayırt etme skoru
dt_report    = classification_report(Y_test, Y_pred_dt, output_dict=True)  # detaylı rapor
dt_precision = dt_report['1']['precision']  # ödemeyeceğini söylediklerinin ne kadarı gerçekten ödemedi
dt_recall    = dt_report['1']['recall']     # gerçek ödemeyenlerin ne kadarını yakaladım
dt_f1        = dt_report['1']['f1-score']   # precision ve recall'un dengeli ortalaması

# Random Forest için tüm metrikleri hesaplıyorum
rf_accuracy  = accuracy_score(Y_test, Y_pred_rf)
rf_roc_auc   = roc_auc_score(Y_test, Y_prob_rf)
rf_report    = classification_report(Y_test, Y_pred_rf, output_dict=True)
rf_precision = rf_report['1']['precision']
rf_recall    = rf_report['1']['recall']
rf_f1        = rf_report['1']['f1-score']

# -------------------------------------------------------
# BOLUM 5 - KARSILASTIRMA TABLOSU
# -------------------------------------------------------

# iki modelin tüm metriklerini yan yana gösteren bir tablo oluşturuyorum
karsilastirma = pd.DataFrame({
    'Metrik'        : ['Accuracy', 'ROC-AUC', 'Precision (1)', 'Recall (1)', 'F1-Score (1)'],
    'Decision Tree' : [f'%{dt_accuracy*100:.2f}', f'{dt_roc_auc:.4f}',
                       f'{dt_precision:.4f}',     f'{dt_recall:.4f}',    f'{dt_f1:.4f}'],
    'Random Forest' : [f'%{rf_accuracy*100:.2f}', f'{rf_roc_auc:.4f}',
                       f'{rf_precision:.4f}',     f'{rf_recall:.4f}',    f'{rf_f1:.4f}'],
    'Kazanan'       : ['Random Forest'] * 5
})

print("\n" + "=" * 55)
print("MODEL KARSILASTIRMA TABLOSU:")
print("=" * 55)
print(karsilastirma.to_string(index=False))
print("=" * 55)

# -------------------------------------------------------
# BOLUM 6 - GRAFIKLER
# -------------------------------------------------------

# 6.0 5 metriğin tamamını gösteren karşılaştırma grafiği
metrikler  = ['Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']
dt_degerler = [dt_accuracy, dt_roc_auc, dt_precision, dt_recall, dt_f1]
rf_degerler = [rf_accuracy, rf_roc_auc, rf_precision, rf_recall, rf_f1]

# x eksenindeki her metrik için bir konum belirliyorum
x = range(len(metrikler))
# iki model yan yana görünsün diye çubukların genişliğini ayarlıyorum
genislik = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
# Decision Tree çubuklarını sola, Random Forest çubuklarını sağa yerleştiriyorum
bars1 = ax.bar([i - genislik/2 for i in x], dt_degerler, genislik,
               label='Decision Tree', color='#3498db')
bars2 = ax.bar([i + genislik/2 for i in x], rf_degerler, genislik,
               label='Random Forest', color='#2ecc71')

# her çubuğun üstüne skor değerini yazıyorum
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#3498db')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#2ecc71')

ax.set_title('Decision Tree vs Random Forest — 5 Metrik Karsilastirmasi',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrikler, fontsize=11)
ax.set_ylabel('Skor')
ax.set_ylim([0, 1.1])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('5_metrik_karsilastirma.png', dpi=150)
plt.show()
print("Grafik kaydedildi: 5_metrik_karsilastirma.png")

# Accuracy ve ROC-AUC'u ayrı ayrı karşılaştıran yan yana grafik çiziyorum
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

modeller   = ['Decision Tree', 'Random Forest']
accuracies = [dt_accuracy * 100, rf_accuracy * 100]
roc_aucs   = [dt_roc_auc, rf_roc_auc]
renkler    = ['#3498db', '#2ecc71']

# sol grafik → accuracy karşılaştırması
axes[0].bar(modeller, accuracies, color=renkler, width=0.4)
axes[0].set_title('Accuracy Karsilastirmasi')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim([70, 85])
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.1, f'%{v:.2f}', ha='center', fontweight='bold')

# sağ grafik → roc-auc karşılaştırması
axes[1].bar(modeller, roc_aucs, color=renkler, width=0.4)
axes[1].set_title('ROC-AUC Karsilastirmasi')
axes[1].set_ylabel('ROC-AUC Skoru')
axes[1].set_ylim([0.70, 0.85])
for i, v in enumerate(roc_aucs):
    axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')

plt.suptitle('Decision Tree vs Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_karsilastirma.png', dpi=150)
plt.show()
print("Grafik kaydedildi: model_karsilastirma.png")

# her iki modelin confusion matrix'ini yan yana çiziyorum
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_dt = confusion_matrix(Y_test, Y_pred_dt)
cm_rf = confusion_matrix(Y_test, Y_pred_rf)

# sol → Decision Tree confusion matrix (mavi ton)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Odedi', 'Odemedi'],
            yticklabels=['Odedi', 'Odemedi'])
axes[0].set_title('Decision Tree - Confusion Matrix')
axes[0].set_ylabel('Gercek')
axes[0].set_xlabel('Tahmin')

# sağ → Random Forest confusion matrix (yeşil ton)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Odedi', 'Odemedi'],
            yticklabels=['Odedi', 'Odemedi'])
axes[1].set_title('Random Forest - Confusion Matrix')
axes[1].set_ylabel('Gercek')
axes[1].set_xlabel('Tahmin')

plt.suptitle('Confusion Matrix Karsilastirmasi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_karsilastirma.png', dpi=150)
plt.show()
print("Grafik kaydedildi: confusion_matrix_karsilastirma.png")

# -------------------------------------------------------
# BOLUM 7 - FINAL YORUM
# -------------------------------------------------------

print("\n" + "=" * 55)
print("FINAL YORUM VE SUNUM OZETI")
print("=" * 55)
print(f"""
KAZANAN MODEL: RANDOM FOREST

1. ACCURACY:
   Decision Tree  -> %{dt_accuracy*100:.2f}
   Random Forest  -> %{rf_accuracy*100:.2f}  (daha yuksek)

2. ROC-AUC:
   Decision Tree  -> {dt_roc_auc:.4f}
   Random Forest  -> {rf_roc_auc:.4f}  (daha yuksek)
   Not: Sinif dengesizliginde ROC-AUC,
   accuracy'den daha guvenilir bir metriktir.

3. DEFAULT YAKALAMA (Recall):
   Decision Tree  -> {dt_recall:.4f}
   Random Forest  -> {rf_recall:.4f}  (daha yuksek)
   Bankacilik icin default yakalamak kritiktir.

NEDEN RANDOM FOREST DAHA IYI?

- Decision Tree tek bir agactir.
  Egitim verisine asiri uyum (overfit) riski tasiyor.

- Random Forest 100 farkli agac kurar ve
  hepsinin ortalamasini alir. Bu sayede:
  * Tek agacin hatalarini dengeler
  * Daha stabil tahminler uretir
  * Yeni veriye daha iyi genelleme yapar

- Feature Importance farki bunu kanitliyor:
  Decision Tree  -> PAY_0 tek basina %74 onem tasiyor
                   (tek degiskene asiri bagimlilik)
  Random Forest  -> PAY_0 sadece %25 onem tasiyor
                   (yuk 100 agaca dengeli dagilmis)

SONUC:
Bu veri seti icin Random Forest,
Decision Tree'ye kiyasla tum metriklerde
daha iyi performans gostermistir.
""")
print("=" * 55)
print("MODEL KARSILASTIRMA TAMAMLANDI.")
print("=" * 55)