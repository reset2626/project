# Conformal Prediction for Time Series 與 KF m1m3／m1m9 閱讀筆記

## 一句話總結

Xu 與 Xie（2023）的核心不是發明新的點預測器，而是用 EnbPI 把任意點預測模型包成可隨時間更新、無須資料 exchangeability、具漸近 coverage 理論的 prediction interval；目前 KF m1m3／m1m9 程式則是「Kalman 分解 + DistPred/ARIMA 樣本分位數區間」，尚未實作 EnbPI 的 out-of-bag residual calibration，因此不能直接套用該論文的 coverage guarantee。

## 1. 論文重點

論文：Chen Xu and Yao Xie, *Conformal Prediction for Time Series*, IEEE TPAMI 45(10), 2023。

### 問題設定

論文假設

\[
Y_t=f(X_t)+\epsilon_t,
\]

目標是對下一期建立窄而可靠的區間。普通 conformal prediction 的有限樣本 marginal coverage 通常依賴 exchangeability，但時間序列明顯具有時間相依，因此論文改以「誤差過程的相依性」與「點預測器估計品質」來控制 coverage gap。

論文同時區分：

- conditional coverage：給定當期特徵後，區間涵蓋真值的機率接近 \(1-\alpha\)；
- marginal coverage：不條件化特徵時的平均涵蓋率接近 \(1-\alpha\)；
- oracle interval：若知道真實 \(f\) 與誤差分布時，可取得的最短有效區間；
- estimated interval：實際由估計模型與過去 residual quantiles 建出的區間。

### EnbPI 的核心流程

1. 從訓練資料 bootstrap 出 \(B\) 組樣本並各自訓練一個點預測模型。
2. 對每個訓練點，只聚合「bootstrap 樣本未包含該點」的模型預測，形成近似 leave-one-out 預測。
3. 計算 out-of-bag/LOO residual \(\hat\epsilon_i=y_i-\hat f_{-i}(x_i)\)。
4. 測試時聚合 ensemble 預測作為區間中心。
5. 在過去 residuals 上選擇長度最短的 \([\beta,1-\alpha+\beta]\) 分位數範圍，允許非對稱區間。
6. 真值陸續回饋後，以 sliding window 更新 residual pool，不必每一期重新訓練模型。

### 理論結果與必要條件

- Assumption 1：短期內誤差可為 i.i.d.；論文也推廣至 linear process 與 strongly mixing errors。
- Assumption 2：LOO 點預測器對真實 \(f\) 的估計誤差 \(\delta_T\) 必須受控制，最好隨樣本數趨近零。
- Theorem 1、2：conditional 與 marginal coverage gap 均由 empirical-CDF 誤差和 \(\delta_T\) 控制，條件成立時漸近為零。
- Theorem 3：估計區間與 oracle interval 的集合差距也會收斂。
- 重要 caveat：EnbPI 實作本身沒有把最壞情況理論上界直接加進區間；保證是漸近且依賴假設，不是無條件的有限樣本精確 conditional coverage。
- batch size 越大、愈久拿不到真值 feedback，coverage 通常會惡化且區間變寬。

### 實證結論

論文在模擬、太陽能、風力、交通資料上比較 EnbPI、AdaptCI、J+aB、QOOB、ICP、weighted ICP 與傳統時序方法。主要結論是 EnbPI 較穩定地達到目標 coverage，區間通常也較短，並可處理 missing data 與 anomaly detection。作者建議 ensemble size \(B\) 約 20–50 通常已足夠。

## 2. meeting 文字資料的脈絡

- `Giordano.txt`：提出 NN-Sieve residual bootstrap，以 ANN 近似非線性 DGP，和 AR-Sieve 比較 prediction interval。M1 是 AR(1)，M3 是 EXPAR(2)，M9 是平滑轉換型非線性模型。強非線性 M3 上 NN-Sieve 明顯優於 AR-Sieve；非線性較弱的 M9，兩者較接近。
- `Babu.txt`：先用 moving-average filter 拆成 trend 與 residual，再以 ARIMA 預測線性／平滑部分、ANN 預測非線性／高頻部分，最後相加。主要指標是 MAE、MSE，不是嚴格的 coverage calibration。
- `senior_thesis.txt`、`senior_thesis_extracted.txt`、`senior_thesis_notes.md`：將 Babu 的 MA filter 換成 Kalman filter，形成 Kalman + ARIMA + ANN + bootstrap。論文區間主要靠 ARIMA interval 與 ANN bootstrap interval 組合，未系統性報告 coverage 與 width；現有 notes 對這項限制的判斷是正確的。
- `dispred.txt`：DistPred 以 proper scoring rule 訓練模型直接輸出 conditional samples，可由樣本分位數形成 probabilistic interval；它是 distribution-free density/forecast learning，但不是 conformal calibration，名詞上的 distribution-free 不等於 coverage-free guarantee。
- `dual_branch_fusion.txt`：DBMS 用 Transformer 與 MLP 雙分支、多尺度分解和自適應 convex fusion改善長期點預測。它可作 EnbPI 的 base predictor，但本身不是區間校準方法，與目前 m1m3/m1m9 的直接關聯較弱。

## 3. KF m1m3／m1m9 程式架構

共讀取五份 notebook：

| 檔案 | DGP | 低頻／高頻預測方式 | 最終區間 |
|---|---|---|---|
| `KF_m1m3joint_lh_distpred.ipynb` | M1 + M3 | Joint DistPred 同時輸出 low/high ensemble | low + high samples 後取 2.5%、97.5% |
| `KF_m1m3Arima_osr_ANN_dispred.ipynb` | M1 + M3 | low：rolling one-step ARIMA；high：ANN DistPred，輸入也含 low ARIMA residual | 組合 samples 後取分位數 |
| `KF_m1m9joint_lh_distpred.ipynb` | M1 + M9 | Joint DistPred 同時預測 low/high | 組合 samples 後取分位數 |
| `KF_m1m9Arima_osr_ANN_dispred.ipynb` | M1 + M9 | low：rolling one-step ARIMA；high：ANN DistPred | 組合 samples 後取分位數 |
| `KF_m1m9Arima_osr.ipynb` | M1 + M9 | 原則上也是 low ARIMA + high ANN；目前內容與另一 ANN notebook 高度重疊 | 組合 samples 後取分位數 |

共同 pipeline：

\[
Y_t=L_t+H_t+\eta_t
\rightarrow \text{simple Kalman filter}
\rightarrow (\widehat L_t,\widehat H_t)
\rightarrow \text{forecast samples}
\rightarrow \text{quantile interval}.
\]

Kalman filter 使用 local-level random-walk 形式，預設 process variance = 0.5、measurement variance = 10.0，並令

\[
\widehat H_t=Y_t-\widehat L_t.
\]

這是平滑／殘差分解，不是從 M1/M3/M9 的真實 state-space equation 推導出的 Kalman model。

### DGP

M1 + M3：

- low：\(L_t=0.6L_{t-1}+e_t\)；
- high：EXPAR(2)，係數隨 \(H_{t-1}^2\) 指數變化；
- high innovation standard deviation 又依賴 \(|L_{t-1}|\)，刻意製造 component volatility dependence；
- 觀測再加標準差 0.15 的 noise。

M1 + M9：

- 程式中的 low 實際為 \(L_t=0.4L_{t-1}+e_t\)，因此嚴格說不再是 Giordano 原始 M1 的 0.6；
- high 使用 logistic gate 的 smooth-transition AR(1)；
- high innovation standard deviation同樣依賴 \(|L_{t-1}|\)。

這兩個 additive DGP 都不是 Giordano 原始單一 M1、M3、M9 的直接重現，而是研究者自行把兩個 component 相加、再加入 cross-component heteroskedasticity 的延伸設計。

### 訓練與評估

- rolling validation 搜尋 window size、ensemble size、epochs、learning rate；
- selection score = 1.25 × nRMSE + 1.50 × |PICP − 0.95| + 0.001 × width；
- 報告 low/high/noisy MSE、RMSE、nRMSE、PICP、mean interval width、CRPS、QICE 與 component correlation；
- prediction interval 是模型輸出樣本的 empirical quantiles，沒有額外 residual calibration。

## 4. 已儲存結果與解讀

notebook 中 h=50、N=300 的已儲存 Monte Carlo 摘要顯示：

| 設定 | noisy MSE | noisy coverage | width |
|---|---:|---:|---:|
| M1+M3 joint | 13.7332 | 95.36% | 18.9676 |
| M1+M9 `Arima_osr` | 3.8036 | 86.24% | 6.4029 |
| M1+M9 ARIMA + ANN DistPred | 3.7887 | 84.80% | 6.1417 |
| M1+M9 joint | 5.4136 | 89.12% | 7.8338 |

解讀：M1+M3 的 nominal 95% coverage 表面上達標，但區間很寬；M1+M9 三種方法都明顯 under-cover。這正是 conformal residual calibration 可補強之處。

但這些數字只能視為 notebook 快照，不能直接當最終比較結果：

- `KF_m1m3Arima_osr_ANN_dispred.ipynb` 的已儲存輸出顯示成「Joint (L,H) DistPred」，和目前 cell source 不一致；
- 多份 notebook 的輸出疑似從其他版本複製或在修改程式後未全部重跑；
- `KF_m1m9Arima_osr.ipynb` 與 `KF_m1m9Arima_osr_ANN_dispred.ipynb` 高度重疊，名稱無法可靠代表實驗差異；
- 因此正式表格前應 restart kernel、run all，並固定 seed、資料、fold 與超參數搜尋預算。

## 5. 論文與目前程式的精確對照

| EnbPI 必要元素 | 目前程式 | 判斷 |
|---|---|---|
| 多個 bootstrap base predictors | DistPred 輸出 ensemble samples，但不一定是 bootstrap 重訓模型 | 概念相似，統計角色不同 |
| out-of-bag/LOO prediction | 未見 | 缺少 |
| OOB residual pool | 未見 | 缺少 |
| residual 的非對稱最短分位數區間 | 直接取預測 samples 的 2.5%/97.5% | 缺少校準 |
| 真值 feedback 後 sliding residual update | rolling forecast 會使用 reference true component | 有 feedback，但不是 EnbPI residual update |
| coverage gap assumptions/diagnostics | 報 PICP，但未檢查 residual mixing 或 estimator consistency | 未驗證 |

最重要的研究結論：DistPred 與 EnbPI 不互斥。DistPred／ARIMA／joint network 可作 base probabilistic or point predictor，EnbPI 再用 OOB residuals 校準最終 \(Y_t\) 區間。實作時應校準「最終觀測序列」的 residual，而不是分別追求 low/high component 各 95% coverage；後者相加後既不保證 95%，也容易造成過寬區間。

## 6. 建議的下一版實驗

1. 先清理五份 notebook，統一函式來源與名稱，全部 restart/run-all，建立可信 baseline。
2. 以現有 KF + joint DistPred 和 KF + ARIMA/ANN 各作一個 base predictor。
3. 對完整訓練序列做 block bootstrap，取得 OOB predictions 與 final-series residuals。
4. 使用 EnbPI 的 \(\beta\) line search，而非固定對稱 2.5%/97.5%。
5. h=1、5、10、25、50 分開評估；若每一步都有 feedback，使用 sliding residual update；若一次預測整段，明確標成 batch/no-feedback。
6. 報告 marginal PICP、30/50 點 sliding coverage、MPIW、coverage-width criterion、CRPS、nRMSE 與 runtime。
7. 比較 raw DistPred quantiles、split conformal、EnbPI 三層結果，才能回答 calibration 是否真正改善 M1+M9 under-coverage。
8. 檢查 OOB residual ACF、Ljung–Box 與 rolling distribution shift，作為 i.i.d./mixing 假設的實證診斷，而不是只報最終 coverage。

## 最終判斷

目前研究程式的主軸更接近「Babu／學長論文的分解式 hybrid forecasting + Giordano 的 nonlinear DGP + DistPred probabilistic samples」，還不是 Xu–Xie 的 conformal time-series method。下一個最有價值的工作不是再增加一種 ANN，而是在現有 base models 外加入正確的 EnbPI residual calibration，並用一致、可重現的 rolling protocol 驗證 M1+M9 的 under-coverage 是否被修正。
