# Babu MA-filter 與相關分解式預測文獻索引

更新日期：2026-08-24

## 判讀方式

- 「有引用 Babu」指正文參考文獻或引用資料庫可確認引用 Babu and Reddy (2014)。
- 「直接延伸」指仍保留「先用 filter／分解器拆開時間序列，再對不同成分建模並合併」的核心流程。
- 「區間預測」必須實際產生上下界或預測分布；文中的 forecast horizon／prediction interval 若只是指預測期間，不算區間估計。
- 引用資料庫可能缺少部分參考文獻，因此「未確認」不等於確定沒有引用。

## 1. 原始研究

### Babu and Reddy (2014)

**A Moving-Average Filter Based Hybrid ARIMA–ANN Model for Forecasting Time Series Data**  
Applied Soft Computing, 23, 27–38. DOI: https://doi.org/10.1016/j.asoc.2014.05.028

- 有無引用 Babu：不適用，這是原始研究。
- 作法：以 moving-average filter 將原序列分為平滑的低波動成分與原序列減去平滑值所得的高波動成分；低波動成分使用 ARIMA，高波動成分使用 ANN，最後加總兩部分預測。
- 輸出：one-step 與 multi-step 點預測。
- 評估：MAE、MSE。
- 區間預測：沒有。

## 2. 直接延續 Babu 類 filter-hybrid 架構，主要做點預測

### Babu and Reddy (2015)

**Prediction of Selected Indian Stock Using a Partitioning–Interpolation Based ARIMA–GARCH Model**  
Applied Computing and Informatics, 11(2), 130–143. DOI: https://doi.org/10.1016/j.aci.2014.09.002

- 有無引用 Babu：有；作者亦為 Babu 與 Reddy，正文引用其 2014 年 MA-filter ARIMA–ANN 方法。
- 作法：先用 MA filter 分成低波動與高波動序列；低波動部分使用 ARIMA，高波動部分經 partitioning 與 interpolation 後使用 GARCH，最後加總預測。
- 目的：提升印度股票資料的多步點預測準確度並維持趨勢。
- 區間預測：沒有，重點是點預測誤差與趨勢保存。

### Babu and Sure (2016)

**Partitioning and Interpolation Based Hybrid ARIMA–ANN Model for Time Series Forecasting**  
Sādhanā, 41(7), 695–706. DOI: https://doi.org/10.1007/s12046-016-0508-5

- 有無引用 Babu：有，且屬於同一研究作者的後續延伸。
- 作法：延續線性／非線性 hybrid 思想，加入 partitioning 與 interpolation，再結合 ARIMA 與 ANN 進行時間序列預測。
- 輸出：點預測。
- 區間預測：未見 coverage、區間寬度或上下界評估。

### Panigrahi, Behera, and Abraham (2017)

**A Fuzzy Filter Based Hybrid ARIMA–ANN Model for Time Series Forecasting**  
DOI: https://doi.org/10.1007/978-3-319-60618-7_58

- 有無引用 Babu：有。
- 作法：以 fuzzy filter 取代固定 MA filter，將序列分成低波動與高波動成分；低波動使用 ARIMA，高波動使用 ANN，最後加總。
- 評估：RMSE、SMAPE、MAE、MASE 等點預測指標。
- 區間預測：沒有。

### Aradhye, Rao, and Mohammed (2019)

**A Novel Hybrid Approach for Time Series Data Forecasting Using Moving Average Filter and ARIMA–SVM**  
DOI: https://doi.org/10.1007/978-981-13-1498-8_33

- 有無引用 Babu：有。
- 作法：保留 MA filter 分解；低波動／線性成分使用 ARIMA，高波動／非線性成分改用 SVM，最後合併預測。
- 評估：MAE、MSE。
- 區間預測：沒有。

### Panigrahi and Behera (2019)

**An Adaptive Fuzzy Filter-Based Hybrid ARIMA–HONN Model for Time Series Forecasting**  
DOI: https://doi.org/10.1007/978-981-10-8055-5_74

- 有無引用 Babu：有。
- 作法：用 adaptive fuzzy filter 分解低、高波動成分；低波動使用 ARIMA，高波動使用 higher-order neural network（HONN／pi-sigma network），最後合併。
- 資料：lynx、sunspot、temperature、passenger、unemployment。
- 區間預測：沒有，公開摘要與結果著重點預測誤差。

### Shahriar et al. (2021)

**Potential of ARIMA-ANN, ARIMA-SVM, DT and CatBoost for Atmospheric PM2.5 Forecasting in Bangladesh**  
Atmosphere, 12(1), 100. DOI: https://doi.org/10.3390/atmos12010100

- 有無引用 Babu：有。
- 作法：建立 moving-average-filter hybrid ARIMA–ANN 與 ARIMA–SVM，並與 decision tree、CatBoost 等方法比較 PM2.5 預測。
- 評估：R²、RMSE、MAE。
- 區間預測：沒有。
- 本資料夾已有開放全文 PDF。

### Huang and Qin (2022)

**Hodrick–Prescott Filter-Based Hybrid ARIMA–SLFNs Model with Residual Decomposition Scheme for Carbon Price Forecasting**  
Applied Soft Computing, 119, 108560. DOI: https://doi.org/10.1016/j.asoc.2022.108560

- 有無引用 Babu：有，引用資料庫將其列為 Babu (2014) 的 citing work。
- 作法：使用 Hodrick–Prescott filter 分離碳價成分，結合 ARIMA-t 與 single-layer feedforward neural networks，並以 residual decomposition 進一步修正誤差。
- 輸出：點預測。
- 區間預測：未見以 coverage 或區間寬度為主要評估的預測區間。

### Musyaffa, Hertono, and Handari (2025)

**Implementation of Moving Average Filter in SARIMA-ANN and SARIMA-SVR Methods for Forecasting Pneumonia Incidence in Jakarta**  
DOI: https://doi.org/10.37905/jjbm.v6i3.30558

- 有無引用 Babu：有。
- 作法：先以 MA filter 分離成分；季節性線性成分使用 SARIMA，其他成分分別使用 ANN 或 SVR。
- 評估：MAPE，並比較不同地區的 SARIMA–ANN 與 SARIMA–SVR。
- 區間預測：沒有。
- 本資料夾已有開放全文 PDF。

## 3. 引用 Babu，且有機率／區間預測

### Yolcu, Jin, and Egrioglu (2016)

**An Ensemble of Single Multiplicative Neuron Models for Probabilistic Prediction**  
DOI: https://doi.org/10.1109/SSCI.2016.7849975

- 有無引用 Babu：有，引用資料庫列為 Babu (2014) 的 citing work。
- 作法：建立 single multiplicative neuron models 的 ensemble，使用 bootstrap 產生預測分布、點預測與 confidence interval。
- 是否延續 Babu filter 分解：否；它是神經網路 ensemble 與 bootstrap，不先用 MA／fuzzy filter 拆成低、高波動。
- 區間預測：有。

### Ding and Meng (2020)

**Point and Interval Forecasting for Wind Speed Based on Linear Component Extraction**  
Applied Soft Computing, 93, 106350. DOI: https://doi.org/10.1016/j.asoc.2020.106350

- 有無引用 Babu：有，原文參考文獻明確列入 Babu and Reddy (2014)。
- 作法：先以 EMD 與 SSA 從風速時間序列抽取線性成分；ARIMA 與 BPNN 負責 deterministic point prediction；ARIMA 與 improved first-order Markov chain（IFOMC）負責 probability interval prediction。
- 是否延續 Babu filter 分解：屬於廣義延伸。它延續「先分解／抽取成分，再以線性與非線性模型預測」的思想，但沒有直接使用 Babu 的 MA filter，也不完全等於低波動 ARIMA＋高波動 ANN。
- 區間預測：有，是目前檢索到最明確的「引用 Babu＋分解＋點與區間預測」正式期刊反例。

### Panja et al. (PARNN, 2022/2024)

**Probabilistic AutoRegressive Neural Networks for Accurate Long-Range Forecasting**  
DOI: https://doi.org/10.1007/978-981-99-8178-6_35

- 有無引用 Babu：有，正文參考文獻列入 Babu and Reddy (2014)。
- 作法：利用 ARIMA feedback error 改善 autoregressive neural network，形成 probabilistic ARNN，並建立 prediction intervals。
- 是否延續 Babu filter 分解：否；屬於 ARIMA 與神經網路 hybrid，但不是先用 filter 分解低、高波動成分。
- 區間預測：有。

### 林裕倫（2024，碩士論文）

**結合卡爾曼濾波器與類神經網路的時間序列預測方法**

- 有無引用 Babu：有，並明確說明由 Babu 的 MA-filter 架構延伸。
- 作法：以 Kalman filter 取代固定 MA filter，分出較平滑與較快速變動成分，分別使用 ARIMA 與 ANN 預測，再以 Sort Bootstrap 建立預測區間。
- 區間預測：有。
- 與本研究關係：是本研究 M5 與後續 M6 發展的重要直接來源。

## 4. 不一定引用 Babu，但同樣是「分解後預測並建立區間」

### Prinzhorn et al. (2024)

**Conformal Time Series Decomposition with Component-Wise Exchangeability**  
arXiv: https://arxiv.org/abs/2406.16766

- 有無引用 Babu：目前未確認；已取得的全文檢索未見 Babu。
- 作法：先分解時間序列，對不同成分分別使用適合的 conformal 方法，再合併各成分的 prediction intervals。
- 區間預測：有。
- 與本研究差異：它校準各分量後再合併；本研究 M6 主要對合併點預測形成的 OOT residual 建立區間。

### Ding et al.／碳價三階段模型（2022）

**A Three-Stage Framework for Vertical Carbon Price Interval Forecast Based on Decomposition–Integration Method**  
DOI: https://doi.org/10.1016/j.asoc.2021.108204

- 有無引用 Babu：目前引用資料未確認。
- 作法：以 ICEEMD 分解，使用 SSA-BPNN 進行點預測，再以 kernel density estimation（KDE）估計誤差分布並形成區間。
- 區間預測：有。

### Zhu et al. (2024/2025)

**Interval Forecasting of Carbon Price With a Novel Hybrid Multiscale Decomposition and Bootstrap Approach**  
Journal of Forecasting, 44(2), 376–390. DOI: https://doi.org/10.1002/for.3199

- 有無引用 Babu：目前引用資料未確認。
- 作法：以 CEEMDAN 將碳價分成多個 modes；對各 mode 執行 bootstrap 並用 XGBoost 預測，再整合為原序列的點與區間預測。
- 評估：預測準確度、interval coverage 與區間寬度。
- 區間預測：有。

### Transformer-based carbon-price model (2023)

**Multi-Step-Ahead and Interval Carbon Price Forecasting Using Transformer-Based Hybrid Model**

- 有無引用 Babu：目前引用資料未確認。
- 作法：Hampel identifier 處理異常值，TVFEMD 分解時間序列，依 sample entropy 重組成分，再以 quantile-loss Transformer 預測各分量並加總。
- 區間預測：有，直接估計不同條件分位數。

### Zhang et al. (2026)

**A Novel Interval Prediction Model Based on LUBE and Decomposition Ensemble for Carbon Price Forecasting and Trading**  
DOI: https://doi.org/10.1007/s44176-026-00067-4

- 有無引用 Babu：目前引用資料未確認。
- 作法：以 VMD 將碳價分成不同頻率成分，再以 LUBE 類模型對各成分直接預測上下界，最後分別加總各成分下界與上界。
- 區間預測：有。

## 5. 綜合判斷

1. Babu (2014) 原始方法只有點預測。
2. 目前找到最直接延續 MA／fuzzy／HP-filter 低高波動 hybrid 架構的多數研究，仍主要評估點預測。
3. 不能寫成「所有引用 Babu 的研究都只有點預測」：Ding and Meng (2020)、PARNN、bootstrap multiplicative-neuron ensemble，以及林裕倫（2024）均提供區間或機率預測。
4. Ding and Meng (2020) 是目前最明確的「引用 Babu、先分解時間序列、同時建立點與區間預測」的正式期刊研究，但其分解與區間方法不同於 Babu，也不同於本研究。
5. 若不限制必須引用 Babu，EMD、ICEEMD、CEEMDAN、TVFEMD、VMD 與 component-wise conformal 等路線已有許多分解式區間預測研究。
6. 因此，本研究不宜宣稱是第一個「分解後做區間預測」的方法。較明確的差異是：causal Kalman decomposition、ARIMA–ANN、依時間順序建立的 OOT residual、自適應 residual window、最短非對稱區間及 Local-scale 的完整組合。

## 6. 論文可採用的保守表述

> Babu與Reddy提出以移動平均濾波器將時間序列分為低波動與高波動成分，再分別使用ARIMA與ANN進行預測。後續多數直接延續此類濾波分解架構的研究，主要著重於點預測準確度；然而，亦有少數研究在引用相關分解式混合模型的基礎上進一步探討機率區間預測。相較之下，本研究採用僅依賴當時及過去資訊的Kalman filter進行因果分解，並由合併點預測所形成的OOT residual建立自適應非對稱預測區間，再透過Local-scale調整不同局部狀態下的殘差尺度。
