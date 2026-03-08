# Ödev-1 · CIFAR-10 k-NN Sınıflandırıcı

> **Derin Sinir Ağları** dersi — k-En Yakın Komşu algoritması ile görüntü sınıflandırma

---

## Proje Hakkında

Bu proje, CIFAR-10 veri seti üzerinde **k-NN (k-En Yakın Komşu)** algoritmasını sıfırdan uygular. Kullanıcı; mesafe metriği (L1 veya L2), k değeri ve test görüntüsü index numarasını seçerek sınıflandırma yapabilir. Sonuçlar Flask tabanlı görsel bir arayüzde gösterilir.

---

## Özellikler

- **L1 (Manhattan)** ve **L2 (Öklid)** mesafe metriği seçimi
- Kullanıcı tanımlı **k değeri**
- 0–9999 arasında **test görüntüsü seçimi**
- Tahmin sonucu, gerçek etiket ve **doğru/yanlış** gösterimi
- **Oy dağılımı** — k komşunun hangi sınıfa oy verdiği
- **En yakın k komşunun görüntüleri** ve mesafeleri
- Kod **fonksiyon kullanmadan düz linear akışla** çalışır

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `knn_cifar10.py` | Veri yükleme, mesafe hesaplama, k-NN algoritması — düz akış |
| `app.py` | Flask sunucusu + görsel web arayüzü |

---

## Kurulum

```bash
pip install flask numpy pillow
```

---

## Veri Seti

CIFAR-10 veri setini resmi siteden indir:

```
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

İndirdikten sonra `cifar-10-batches-py` klasörünü proje dizinine koy.

Klasör yapısı şöyle olmalı:

```
cifar-10-batches-py/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```

---

## Çalıştırma

`app.py` içindeki `DATA_PATH` satırını kendi bilgisayarına göre güncelle:

```python
DATA_PATH = r"C:\Users\KULLANICI_ADI\Downloads\cifar-10-python\cifar-10-batches-py"
```

Sonra terminalde:

```bash
python app.py
```

Tarayıcıda aç → [http://localhost:5000](http://localhost:5000)

---

## Nasıl Kullanılır

1. **L1** (Manhattan) veya **L2** (Öklid) mesafe metriğini seç
2. **k değeri** gir — kaç komşuya bakılacağını belirler
3. **Test index** gir (0–9999) — hangi görüntünün sınıflandırılacağını seçer
4. **Sınıflandır** butonuna bas
5. Sonuçta tahmin, gerçek etiket, oy dağılımı ve komşu görüntüler görünür

---

## Ekran Görüntüleri

### L2 · Doğru Tahmin — Airplane (index: 90, k: 5)
<img width="954" height="1370" alt="Ekran görüntüsü 2026-03-08 033253" src="https://github.com/user-attachments/assets/8cc80169-049d-4c6c-8729-9cffff05be46" />


### L1 · Doğru Tahmin — Bird (index: 900, k: 5)
<img width="906" height="1380" alt="Ekran görüntüsü 2026-03-08 033412" src="https://github.com/user-attachments/assets/19b0a38e-e8b8-46d5-91f1-f86f26a9600e" />


### L1 · Yanlış Tahmin — Cat → Bird (index: 0, k: 9)
<img width="693" height="1294" alt="Ekran görüntüsü 2026-03-08 033329" src="https://github.com/user-attachments/assets/9728cbeb-fa54-4735-b65f-d0dfd4eeb1f6" />


> k-NN'in doğruluk oranı CIFAR-10'da yaklaşık %30-40 civarındadır. Kedi-köpek-kuş gibi benzer görünen sınıflar arasında yanlış tahminler normaldir.

---

## Algoritma Detayları

### L1 — Manhattan Mesafesi
Her pikselin farkının mutlak değerlerini toplar.

```
d(x, y) = Σ |xᵢ - yᵢ|
```

### L2 — Öklid Mesafesi
Farkların karelerini alıp toplar, karekök çeker.

```
d(x, y) = √Σ (xᵢ - yᵢ)²
```

### k-NN Sınıflandırma
1. Sorgu görüntüsü ile tüm 50.000 eğitim görüntüsü arasındaki mesafe hesaplanır
2. En küçük k mesafeye sahip komşular seçilir
3. Bu k komşu arasında en çok hangi sınıf varsa o tahmin edilir (çoğunluk oylaması)
