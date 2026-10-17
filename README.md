# El Yazısı Rakamlarının Sınıflandırılması

**PDF raporları:** [Rapor 1](raporlar/Rapor_1_Ogrenme_ve_Deney_Analizi.pdf) · [Rapor 2](raporlar/Rapor_2_Akademik_Makale.pdf)

Bu depo, scikit-learn Digits veri kümesindeki 0–9 rakamlarını farklı sınıflandırıcılarla karşılaştıran Yapay Sinir Ağları F26 dönem ödevinin kodlarını, raporlarını ve kayıtlı sonuçlarını içerir.

**Öğrenci:** Reyhan Onbaşı  
**Öğrenci numarası:** 202612696008  
**Veri kümesi:** scikit-learn Digits (1.797 örnek, 8×8 piksel, 10 sınıf)

## Kapsam

Beş klasik algoritma ailesi (lojistik regresyon, KNN, Random Forest, RBF-SVM, Gradient Boosting) ve MLP-A, MLP-B, CNN-C ağ yapılandırmaları karşılaştırılmıştır. Eğitim/doğrulama/test ayrımı 1.257/270/270 örnektir. Klasik modellerde çapraz doğrulama ve eğitim üzerinde fit edilen ön işleme kullanılır. Model seçimi doğrulama makro F1 üzerinden yapılmıştır.

## Ana deney sonuçları

| Model | Doğrulama doğruluğu | Test doğruluğu | Test makro F1 |
| --- | ---: | ---: | ---: |
| CNN-C | %99,26 | %98,15 | 0,98124 |
| Logistic Regression | %97,78 | %98,52 | 0,98516 |
| SVM (RBF) | %97,41 | %98,89 | 0,98885 |
| Random Forest | %96,67 | %97,41 | 0,97379 |
| KNN | %96,30 | %98,15 | 0,98146 |
| Gradient Boosting | %95,19 | %96,30 | 0,96271 |

CNN-C doğrulama makro F1 ile seçilmiştir. Testte daha yüksek sonuç veren SVM, model seçimini geriye dönük değiştirmek için kullanılmamıştır. Kesin McNemar testi (p=0,6875) bu testte istatistiksel üstünlük için yeterli kanıt göstermemiştir.

## Depo yapısı

- `notebooks/Digits_Model_Karsilastirmasi.ipynb`: klasik sınıflandırıcılar ile MLP ve CNN modellerinin karşılaştırılması.
- `notebooks/CNN_Kararlilik_Analizi.ipynb`: CNN modelinin beş farklı rastgele başlangıçla kararlılık analizi.
- `notebooks/CNN_Mimari_Karsilastirmasi.ipynb`: CNN mimarisi ve düzenlileştirme ayarlarının karşılaştırılması.
- `raporlar/`: iki raporun PDF ve DOCX kopyaları.
- `sekiller/`: raporlarda kullanılan yüksek çözünürlüklü şekiller.
- `sonuclar/ana_deney/`: kaydedilmiş ana deney metrikleri ve çalışma özeti.
- `sonuclar/kararlilik/`: beş tohumlu deneyin özetleri.
- `sonuclar/ek_dogrulama/`: üç yapılandırma × üç tohum ek analizinin doğrulama ölçütleri.

## Çalıştırma

Google Colab'da `notebooks/Digits_Model_Karsilastirmasi.ipynb` dosyasını açıp hücreleri yukarıdan aşağı çalıştırabilirsiniz. Orijinal deneyde kullanılan paket sürümleri `requirements.txt` ve `sonuclar/ana_deney/environment.json` içinde kaydedilmiştir. TensorFlow, Python ve donanım farkları nedeniyle yeni çalıştırmanın bazı ölçütleri kayıtlı deneyden farklı olabilir. Yeni çalıştırma, `sonuclar/ana_deney/` dizinindeki geçmiş kaydı otomatik olarak değiştirmez.

Ek analizler, ana deneyde model seçiminin yerine geçmez; aynı veri ayrımı üzerinde sonradan yürütülen keşifsel değerlendirmelerdir.

## Veri erişimi

- scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- Özgün UCI koleksiyonu: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

Veri kümesi `sklearn.datasets.load_digits()` ile yerel olarak yüklenir; ayrı bir veri dosyasına ihtiyaç yoktur.

## Yöntem, kaynaklar ve yapay zekâ kullanımı

Akademik kaynaklar, yöntem açıklamaları, sınırlılıklar ve kullanım beyanı `raporlar/` dizinindeki raporlarda bulunur. Kod düzenleme ve kontrol süreçlerinde üretken yapay zekâ desteği kullanılmıştır; deney sonuçlarının kaynağı kaydedilen çalışma dosyalarıdır. Kaynakça ve son rapor sürümleri teslim öncesinde ayrıca doğrulanmalıdır.

## Not

Depoda kodlar, raporlar ve deneylerin sayısal sonuçları yer almaktadır. Eğitilmiş model ağırlıkları ve büyük boyutlu ikili dosyalar depoya eklenmemiştir.
