# El Yazısı Rakamlarının Sınıflandırılması — YSA F26

**Ders projesi:** Yapay Sinir Ağları F26  
**Öğrenci:** Reyhan Onbaşı | **No:** 202612696008  
**Kaynak:** scikit-learn Digits, 1.797 adet 8 × 8 rakam görüntüsü

**Raporlar:** [Rapor 1 (PDF)](raporlar/Rapor_1_Ogrenme_ve_Deney_Analizi.pdf) · [Rapor 2 (PDF)](raporlar/Rapor_2_Akademik_Makale.pdf)

## Son Colab çalışma kaydı

Güncel deneyler, 17 Ekim 2026 tarihli **tek Google Colab oturumunda** gerçekleştirilmiştir (`20261017T160921Z`).

- [Çıktılı Colab notebook'u](notebooks/202612696008_Reyhan_Onbasi_YSA_F26_Tum_Deneyler.ipynb)
- [Ana deney performansları](sonuclar/colab_20261017/ana_deney/table_6_final_performance.csv)
- [CNN-C beş başlangıç denemesi](sonuclar/colab_20261017/kararlilik/all_five_runs.csv)
- [Ek doğrulama](sonuclar/colab_20261017/ek_dogrulama/validation_summary.csv)
- [Ortam sürümleri](sonuclar/colab_20261017/environment.json)

**Veri bölünmesi:** eğitim 1.257, doğrulama 270, test 270. Ön işleme sırasında öğrenilen istatistikler yalnızca eğitim bölünmesi içinde öğrenilmiştir. Ana model, test kümesine bakılmadan doğrulama **makro F1** ile seçilmiştir.

| Algoritma | Doğrulama doğruluğu | Test doğruluğu | Test makro F1 |
| --- | ---: | ---: | ---: |
| CNN-C (doğrulama ile seçilen) | %99,26 | %98,15 | 0,98124 |
| SVM (RBF) | %97,41 | %98,89 | 0,98885 |
| Logistic Regression | %97,78 | %98,52 | 0,98516 |
| KNN | %96,30 | %98,15 | 0,98146 |
| Random Forest | %96,67 | %97,41 | 0,97379 |
| Gradient Boosting | %95,19 | %96,30 | 0,96271 |

CNN-SVM eşleştirilmiş **kesin McNemar p=0,6875**: istatistiksel üstünlük kanıtı yeterli değildir. CNN-C'nin 101, 202, 303, 404, 505 tohumlarıyla elde edilen **beş koşuluk ortalama test doğruluğu %98,74 (örnek s.s. 1,13 yüzde puan)**. Kararlılık ve ek yapılandırma denemeleri aynı test gözlemlerini tekrar kullanmıştır; yeni ve bağımsız bir dış test olarak yorumlanmaz.

## Depo düzeni

- `notebooks/`: son çıktı alınmış birleşik Colab notebook'u; eski deney defterleri geçmiş çalışma sürümleri olarak korunabilir.
- `raporlar/`: son Colab sonuçlarıyla eşleşen iki raporun DOCX/PDF dosyaları.
- `sekiller/`: son Colab sonuçlarıyla güncellenmiş 600 dpi grafikler.
- `sonuclar/colab_20261017/`: CSV/JSON özetleri, eğitim planı ve sürümler.
- `sonuclar/ana_deney/`, `sonuclar/kararlilik/`, `sonuclar/ek_dogrulama/`: depoda daha önce bulunan *eski deney kayıtları*; tarihleri korunmalıdır.

Büyük boyutlu model ağırlıkları, tüm tahmin olasılıkları ve ham eğitim geçmişleri MS Teams teslimine eklenen **`YSA_F26_Tum_Deneyler_Colab_20261017T160921Z.zip`** içinde arşivlenmiştir; ikili model dosyaları GitHub'a eklenmeyerek depodaki boyut sınırlanmıştır.

## Veri, üretilebilirlik ve kullanım notları

- Veri erişim bağlantısı: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- UCI özgün koleksiyonu: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Kod, Google Colab'a yüklenerek hücre sırasıyla çalıştırılabilir; versiyon ve donanım farkları yeni eğitimlerde sonuçları değiştirebilir.
- Üretken yapay zekâ kod gözden geçirme, hata giderme ve metin düzenleme süreçlerinde kullanılmıştır; sayısal değerler kaydedilen gerçek eğitim çıktılarından gelmektedir. Akademik sorumluluk ve kaynak kullanımı öğrencidedir.
