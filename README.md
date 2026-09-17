# 🏠 House Prices — Advanced Regression Techniques

[Kaggle yarışması](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) için uçtan uca, sızıntısız bir scikit-learn pipeline çözümü.
Metrik: **RMSLE** (log fiyat uzayında RMSE).

## Sonuçlar

| Aşama | CV RMSLE (5-kat) |
|---|---|
| Baseline — Decision Tree (34 OHE + sayısal, ham hedef) | 0.2192 |
| Ordinal + nominal ön işleme, Ridge, ham hedef | 0.1855 |
| + log hedef (log1p) | 0.1452 |
| + özellik mühendisliği & aykırı gözlem temizliği (en iyi tek model: SVR) | 0.1103 |
| **8 modelin NNLS ağırlıklı blend'i** | **0.1062** |

Baseline'a göre %52 hata azalması.

## Yaklaşım

1. **EDA** — eksik değer profili, hedef çarpıklığı, `GrLivArea` aykırı gözlemleri.
2. **Baseline** — tip bazlı ön işleme (impute + MinMax + OHE) ve Decision Tree; özel RMSLE scorer ile 5-kat CV.
3. **Ön işleme iterasyonları**
   - NaN = "yapı yok" yorumu (`BsmtQual`, `GarageType`, `FireplaceQu`, ...)
   - 20 özellik için **ordinal kodlama** (`Po < Fa < TA < Gd < Ex`), kalanlar için `min_frequency`'li one-hot
   - çarpık sayısal özelliklere `log1p` + `RobustScaler`
   - türetilmiş özellikler: `TotalSF`, `TotalBath`, `TotalPorchSF`, `HouseAge`, `RemodAge`, `QualLivArea`, ...
   - `MoSold` için **döngüsel** (sin/cos) kodlama
   - istatistiksel özellik seçimi denemeleri: mutual information, L1, varyans, Spearman korelasyon filtresi
   - **log hedef** (`log1p(SalePrice)`) — RMSLE ile birebir uyumlu
4. **Model karşılaştırması** — Ridge, Lasso, ElasticNet, SVR, GradientBoosting, XGBoost, LightGBM, CatBoost; her biri için 5-kat OOF tahmin, skor ve süre.
5. **Ensemble** — OOF tahminler üzerinde NNLS ile öğrenilen ağırlıklı blend + Kaggle submission.

Tüm dönüşümler `Pipeline` / `ColumnTransformer` içinde; istatistikler yalnızca eğitim katmanından öğrenilir (veri sızıntısı yok). `SEED = 42`, `Restart & Run All` ~8 dakikada tekrar üretilebilir.

## Kurulum

```bash
pip install -r requirements.txt

# Veri setleri
curl https://d32aokrjazspmn.cloudfront.net/materials/houses-train-raw.csv > data/train.csv
curl https://d32aokrjazspmn.cloudfront.net/materials/houses_test_raw.csv > data/test.csv
curl https://d32aokrjazspmn.cloudfront.net/materials/houses_sample_submission.csv > data/sample_submission.csv
```

Ardından `houses_kaggle_competition.ipynb` dosyasını açıp çalıştırın. Notebook veri klasörünü otomatik bulur: lokalde `data/`, Kaggle'da `/kaggle/input/...`.

Testler: `make` veya `pytest -v`.

## Kaggle notebook (English)

Bu depodaki ikinci notebook, yarışma için hazırlanmış bağımsız ve İngilizce sürümdür:
[`notebooks/house-prices-advanced-pipeline.ipynb`](notebooks/house-prices-advanced-pipeline.ipynb) →
[Kaggle'da yayında](https://www.kaggle.com/code/senanuretin/house-prices-pipeline-and-blending)

Bootcamp notebook'undan farkları: Optuna ile ayarlanmış 9 model, tekrarlı K-fold doğrulama,
hata analizi, permutation importance ve **CV ile public leaderboard'un neden çeliştiğini**
ölçümlerle anlatan bir bölüm.

### Gönderim geçmişi (public LB)

| Gönderim | CV RMSLE | Public LB |
|---|---|---|
| 8 modelli NNLS blend (tuning öncesi) | 0.1062 | **0.11982** |
| 17 modelli ortalama (tuned + untuned) | 0.1068 | 0.12001 |
| 9 tuned model, eşit ağırlıklı ortalama | 0.1065 | 0.12037 |
| 9 tuned model, NNLS ağırlıkları | 0.1056 | 0.12080 |

En iyi CV'ye sahip varyantın LB'si en kötü: Optuna CV katlarına, NNLS de OOF matrisine aşırı uyuyor.
Dört varyant arasındaki 0.001'lik fark 1.459 satırlık public LB'nin gürültü bandının içinde kaldığı için
final gönderim, hiçbir değerlendirme kümesine ağırlık uydurmayan eşit ağırlıklı ortalamadır.

## Çıktılar

- `data/submission_baseline.csv` — baseline tahminleri
- `data/submission_final.csv` — bootcamp notebook'unun blend gönderimi
- `data/submission.csv` — İngilizce notebook'un eşit ağırlıklı blend gönderimi
