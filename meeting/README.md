# Kalman Filter Forecasting

本專案研究 Kalman Filter、EnbPI、ARIMA 與神經網路在時間序列預測及預測區間上的應用。

## 專案結構

- `notebooks/experiments/0050/`：台灣 0050 資料實驗
  - `dispred/`：分布與預測區間方法
  - `EnbPI/`：EnbPI 方法
- `notebooks/experiments/m1m3/`：m1m3 模擬資料實驗
  - `dispred/`：分布與預測區間方法
  - `EnbPI/`：EnbPI 與 Oracle 比較
- `notebooks/experiments/m1m9/`：m1m9 模擬資料實驗
  - `dispred/`：分布與預測區間方法
  - `EnbPI/`：EnbPI 與 Oracle 比較
- `notebooks/exploration/`：論文重現與探索性 notebook
- `notebooks/archive/`：不再使用但暫時保留的舊實驗
- `src/kf_forecasting/models/`：可重複使用的模型程式
- `docs/papers/`：論文與相關文字資料
- `docs/notes/`：研究筆記
- `data/`：原始及處理後資料
- `results/`：預測結果、指標與圖表
- `tests/`：自動化測試

## 環境安裝

在本資料夾執行：

```powershell
python -m pip install -e .