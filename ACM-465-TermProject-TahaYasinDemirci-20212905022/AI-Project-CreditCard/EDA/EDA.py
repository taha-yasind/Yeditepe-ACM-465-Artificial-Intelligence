# ============================================================
# KREDİ KARTI DEFAULT TAHMİNİ
# ============================================================
# Amaç   : Müşterilerin bir sonraki ay kredi kartı borcunu
#           ödeyip ödeyemeyeceğini tahmin etmek.
# Dosya  : eda.py → Keşifsel Veri Analizi (EDA)
# Veri   : UCI Credit Card Default Dataset (30.000 müşteri)
# ============================================================

# numpy: sayısal hesaplamalar için kullanıyorum (matematiksel işlemler)
import numpy as np
# pandas: tablo şeklindeki verileri okumak ve işlemek için kullanıyorum
import pandas as pd
# matplotlib: grafik çizmek için kullanıyorum
import matplotlib.pyplot as plt
# seaborn: daha şık grafikler çizmek için kullanıyorum (matplotlib üzerine kurulu)
import seaborn as sns

# veri setini Excel dosyasından okuyorum ve df adlı değişkene kaydediyorum
df = pd.read_excel('../creditCard_UCI.xlsx')

# -------------------------------------------------------
# 1.1 GENEL BAKIŞ
# -------------------------------------------------------

print("=" * 50)
print("VERİ SETİ BOYUTU:")
# df.shape → tablonun kaç satır ve kaç sütundan oluştuğunu gösteriyor
print(df.shape)

print("\nSÜTUN İSİMLERİ VE VERİ TİPLERİ:")
# df.dtypes → her sütunun ismini ve veri tipini (sayı mı, metin mi?) gösteriyor
print(df.dtypes)

print("\nİLK 5 SATIR:")
# df.head() → tablonun ilk 5 satırını gösteriyor, veriye hızlı bakış için kullanıyorum
print(df.head())

# -------------------------------------------------------
# 1.2 EKSİK VERİ KONTROLÜ
# -------------------------------------------------------

print("\n" + "=" * 50)
print("EKSİK VERİ KONTROLÜ:")
# isnull() → her hücrenin boş olup olmadığını kontrol ediyor
# .sum().sum() → önce sütun sütun, sonra toplam boş hücre sayısını hesaplıyor
print("Toplam eksik değer sayısı:", df.isnull().sum().sum())
# .values.any() → herhangi bir boş hücre var mı? True/False döndürüyor
print("Eksik veri var mı?", df.isnull().values.any())

# -------------------------------------------------------
# 1.3 TANIMLAYICI İSTATİSTİKLER
# -------------------------------------------------------

print("\n" + "=" * 50)
print("TANIMLAYICI İSTATİSTİKLER:")

# ID ve hedef değişkeni dışındaki tüm sayısal sütunları seçiyorum
num_var = [col for col in df.columns if col not in ['ID', 'default.payment.next.month']]
# .describe() → ortalama, min, max, standart sapma gibi özet istatistikleri hesaplıyor
# .T → tabloyu döndürüyorum (satırları sütun, sütunları satır yapıyorum) okunması kolaylaşsın diye
desc_summ = df[num_var].describe().T
print(desc_summ)

# -------------------------------------------------------
# 1.4 TARGET DEĞİŞKENİ İNCELEMESİ
# -------------------------------------------------------

print("\n" + "=" * 50)
print("TARGET DAĞILIMI (default.payment.next.month):")
# value_counts() → hedef sütundaki her değerin kaç kez geçtiğini sayıyor
# 0 = ödedi, 1 = ödemedi
target_counts = df['default.payment.next.month'].value_counts()
print(target_counts)
print("\nYüzdelik dağılım:")
# normalize=True → sayı yerine oran döndürüyor | .round(3)*100 → yüzdeye çeviriyorum
print(df['default.payment.next.month'].value_counts(normalize=True).round(3) * 100)

# target dağılımı grafiği
plt.figure(figsize=(6, 4))
# bar chart çiziyorum: x ekseni etiketler, y ekseni müşteri sayıları, renkler yeşil/kırmızı
plt.bar(['Ödedi (0)', 'Ödemedi (1)'],
        target_counts.values,
        color=['#2ecc71', '#e74c3c'])
plt.title('Target Dağılımı: Ödedi mi / Ödemedi mi?')
plt.ylabel('Müşteri Sayısı')
# her çubuğun üstüne sayıyı yazıyorum
for i, v in enumerate(target_counts.values):
    plt.text(i, v + 200, str(v), ha='center', fontweight='bold')
plt.tight_layout()
# grafiği dosyaya kaydediyorum
plt.savefig('target_dagilimi.png', dpi=150)
plt.show()
print("→ Grafik kaydedildi: target_dagilimi.png")

# -------------------------------------------------------
# 1.5 SAYISAL DEĞİŞKENLERİN DAĞILIMI
# -------------------------------------------------------

# belirli bir sütun için özet istatistik ve histogram çizen fonksiyon tanımlıyorum
def num_summary(data, numerical_col, plot=False):
    # incelemek istediğim yüzdelik dilimler
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(f"\n--- {numerical_col} ---")
    # seçilen sütunun özet istatistiklerini bu yüzdelik dilimlere göre yazdırıyorum
    print(data[numerical_col].describe(quantiles).T)
    if plot:
        # hist() → o sütunun dağılımını histogram olarak çiziyor
        data[numerical_col].hist(bins=30, color='steelblue', edgecolor='white')
        plt.xlabel(numerical_col)
        plt.title(f'{numerical_col} Dağılımı')
        plt.tight_layout()
        plt.show(block=True)

# kredi limiti ve yaş sütunlarının dağılımına bakıyorum
num_summary(df, 'LIMIT_BAL', plot=True)
num_summary(df, 'AGE', plot=True)

# -------------------------------------------------------
# 1.6 TARGET'A GÖRE DEĞİŞKEN ANALİZİ
# -------------------------------------------------------

print("\n" + "=" * 50)
print("TARGET'A GÖRE ORTALAMA KREDİ LİMİTİ:")
# groupby → ödeyenler ve ödemeyenler olarak grupluyorum
# ardından her grubun ortalama kredi limitini hesaplıyorum
print(df.groupby('default.payment.next.month')['LIMIT_BAL'].mean())

print("\nTARGET'A GÖRE ORTALAMA YAŞ:")
# aynı mantıkla gruba göre ortalama yaşı hesaplıyorum
print(df.groupby('default.payment.next.month')['AGE'].mean())

# belirli bir sayısal sütunun gruba göre ortalamasını yazdıran yardımcı fonksiyon
def target_summary_with_num(dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col: 'mean'}), end='\n\n')

print("\nTARGET'A GÖRE ORTALAMA FATURA TUTARLARI:")
# son 3 ayın fatura tutarlarını (BILL_AMT1, 2, 3) gruba göre karşılaştırıyorum
for col in ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']:
    target_summary_with_num(df, 'default.payment.next.month', col)

# -------------------------------------------------------
# 1.7 KORELASYON ISI HARİTASI
# -------------------------------------------------------

print("\n" + "=" * 50)
print("KORELASYON ANALİZİ yapılıyor...")

# ID sütununu çıkarıp geri kalan sütunlar arasındaki korelasyonu hesaplıyorum
# korelasyon: iki değişkenin birlikte nasıl değiştiğini gösteriyor (-1 ile 1 arası)
corr = df.drop(columns=['ID']).corr()

plt.figure(figsize=(14, 10))
# heatmap → korelasyon matrisini renkli ısı haritası olarak görselleştiriyorum
# RdBu renk paleti: mavi=negatif korelasyon, kırmızı=pozitif korelasyon
# center=0 → ortanın beyaz olmasını sağlıyor
sns.heatmap(corr, cmap='RdBu', center=0, annot=False, linewidths=0.5)
plt.title('Değişkenler Arası Korelasyon Isı Haritası')
plt.tight_layout()
plt.savefig('korelasyon_haritasi.png', dpi=150)
plt.show()
print("→ Grafik kaydedildi: korelasyon_haritasi.png")

# -------------------------------------------------------
# 1.8 EDA ÖZET SONUÇLARI
# -------------------------------------------------------

print("\n" + "=" * 50)
print("EDA SONUÇLARI ÖZET:")
print(f"  • Toplam müşteri sayısı : {df.shape[0]:,}")
print(f"  • Toplam değişken sayısı: {df.shape[1]}")
print(f"  • Eksik veri            : YOK")
print(f"  • Ödeyenler  (0)        : {target_counts[0]:,} (%{target_counts[0]/len(df)*100:.1f})")
print(f"  • Ödemeyenler (1)       : {target_counts[1]:,} (%{target_counts[1]/len(df)*100:.1f})")
print(f"  • Sınıf dengesizliği    : VAR → modelde dikkate alınacak")
print("=" * 50)
print("EDA TAMAMLANDI.")