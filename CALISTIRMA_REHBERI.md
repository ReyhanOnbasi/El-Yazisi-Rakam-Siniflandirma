# YSA F26 — Deneyleri açma ve sonuçları denetleme

Bu çalışma scikit-learn Digits (1.797 görüntü; 10 sınıf) veri kümesi üzerinde yürütülmüştür.
Ana deneyin ve ek deneylerin gerçek çalıştırma sonuçları `sonuclar/` altındaki ZIP arşivinde
bulunur. Yeniden çalıştırma, rastgelelik ve ortam farkları nedeniyle birebir aynı sayıları
üretmeyebilir. Arşivdeki sonuçlar yeniden eğitim yapılarak değiştirilmemiştir.

## Google Colab bağlantısı

- [Çalıştırılmış ana notebook'u GitHub'da aç](https://github.com/ReyhanOnbasi/El-Yazisi-Rakam-Siniflandirma/blob/main/notebooks/202612696008_Reyhan_Onbasi_YSA_F26_Tum_Deneyler.ipynb)
- [Notebook'u doğrudan Google Colab'da aç](https://colab.research.google.com/github/ReyhanOnbasi/El-Yazisi-Rakam-Siniflandirma/blob/main/notebooks/202612696008_Reyhan_Onbasi_YSA_F26_Tum_Deneyler.ipynb)
- [Veri kaynağı: sklearn.datasets.load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

## Nasıl incelenir?

1. GitHub veya Colab bağlantısından notebook'u açın. Kodların önceden kaydedilmiş
   `execution_count` ve çıktı hücreleri, daha önce yapılmış çalıştırmayı gösterir.
2. Yeniden üretim istenirse Colab'da **Çalışma Zamanı > Tümünü Çalıştır** seçilir.
   GPU gerekli olmamakla birlikte sinir ağı hücrelerinin tamamı eğitim gerçekleştirir;
   çalışma süresi Colab ortamına göre değişir.
3. **Bölüm A**: Digits veri keşfi, 1.257/270/270 eğitim/doğrulama/test ayrımı,
   Logistic Regression, KNN, Random Forest, RBF-SVM, Gradient Boosting ve MLP/CNN
   karşılaştırması. Klasik modellerde `Pipeline` ve `GridSearchCV` yalnızca eğitim
   verisiyle kurulur; test verisi hiperparametre seçimine katılmaz.
4. **Bölüm B**: CNN-C için beş başlangıç tohumu (`101, 202, 303, 404, 505`).
5. **Bölüm C**: Üç CNN yapılandırmasının üçer başlangıçla **yalnızca doğrulama**
   veri kümesinde karşılaştırılması; test sonuçları bu seçimde kullanılmaz.
6. **Bölüm D**: Ortam bilgileri ve birleşik ZIP çıktısının oluşturulması.

## Sonuçların kontrol edilmesi

- Notebook: `notebooks/202612696008_Reyhan_Onbasi_YSA_F26_Tum_Deneyler.ipynb`
- Kaydedilmiş Colab çıktı arşivi: `sonuclar/YSA_F26_Tum_Deneyler_Colab_20261017T160921Z.zip`
- Arşivde ana sonuçlar `table_6_final_performance.csv`, başlangıç denemeleri
  `all_five_runs.csv`, ek mimari doğrulama sonuçları `validation_summary.csv` içinde yer alır.
- Veri ayırma indisleri: `split_indices.npz`.
- `DOSYA_SHA256.json`, teslim paketindeki dosyaların boyut ve SHA-256 kontrolünü sağlar.

**Önemli:** Notebook'un yeniden çalıştırılması arşivlenmiş özgün test sonuçlarının yerine
geçmez. Bu çalışmada test skorunu gördükten sonra model seçimi geriye dönük değiştirilmez.
