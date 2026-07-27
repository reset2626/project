# Prinzhorn et al. (2024)：分解式 Conformal Time Series Prediction 閱讀筆記

## 1. 文獻資訊

- Derck W. E. Prinzhorn, Thijmen Nijdam, Putri A. van der Linden, Alexander Timans.
- **Conformal time series decomposition with component-wise exchangeability**.
- *Proceedings of the Thirteenth Symposium on Conformal and Probabilistic Prediction with Applications*, PMLR 230, pp. 432–465, 2024.
- 官方頁面：<https://proceedings.mlr.press/v230/prinzhorn24a.html>
- 官方程式：<https://github.com/dweprinz/CP-TSD>
- 本地 PDF：`docs/notes/prinzhorn24a_conformal_time_series_decomposition.pdf`

## 2. 一句話摘要

這篇論文先將原始時間序列分解成 trend、seasonality 與 remainder，再依各成分不同的時間相依性，分別套用適合的 conformal prediction 方法，最後把三個 component prediction intervals 的下界相加、上界相加，重組成原始時間序列的 prediction interval。

它和目前 KF–ARIMA–ANN–EnbPI 方法在「先分解、分流預測、各成分都有區間」這一層非常接近；最關鍵的差異在於 final interval 的建立方式。

## 3. 論文想解決的問題

一般 conformal prediction 的有限樣本 coverage guarantee 通常依賴 exchangeability，但時間序列具有 serial dependence，不能任意交換時間順序。作者認為，原始時間序列雖然整體不具 exchangeability，分解後的不同成分卻可能具有不同程度的 exchangeability：

- trend 有很強的時間相依，因此視為 non-exchangeable；
- seasonality 只在相似週期位置之間近似可交換，因此視為 locally exchangeable；
- remainder 若分解良好，應接近白噪音，因此可近似視為 globally exchangeable。

作者的主張不是「所有成分都套同一種 conformal 方法」，而是針對每個成分的統計結構選擇不同的 calibration 方法。

## 4. 完整流程

論文假設 additive decomposition：

\[
Y_t=T_t+S_t+R_t,
\]

其中 (T_t) 是 trend、(S_t) 是 seasonal component、(R_t) 是 remainder。

流程如下：

```text
原始時間序列 Y_t
        ↓
使用 STL 分解
        ↓
Trend T_t ─────→ EnbPI 或 ACI
Season S_t ────→ BinaryPoint / BinaryLocal / ExpLocal
Remainder R_t ─→ CV+（也比較 EnbPI、ACI）
        ↓
每個 component 取得自己的 point forecast 與 conformal interval
        ↓
component lower bounds 相加、upper bounds 相加
        ↓
得到原始 Y_t 的 recomposed prediction interval
```

他們使用 autoregressive features 訓練 predictor，也就是以先前幾期資料預測下一期。比較的 predictor 包含 linear regression、MLP 與 gradient boosting。整個實驗模擬 sequential forecasting，每次收到新的真值後再往下一期預測。

## 5. 每個成分使用什麼 conformal 方法

### 5.1 Trend：non-exchangeable

Trend 具有強烈 serial correlation，不能假設任意時間點可交換，因此作者採用為時間序列設計的：

- EnbPI；
- Adaptive Conformal Inference（ACI）。

EnbPI 透過 bootstrap ensemble 形成 leave-one-out scores，並持續更新 calibration residual set。作者引用的 EnbPI 理論條件包括 stationary、strongly mixing error process，coverage 屬於有限樣本誤差界與漸近 coverage，而不是一般 i.i.d. conformal 的直接保證。

### 5.2 Seasonality：locally exchangeable

作者認為固定週期中的相同位置，例如每週一或每天尖峰時刻，彼此可能近似 exchangeable，因此提出三種 weighted conformal 方法：

- **BinaryPoint**：只使用歷史週期中和目前預測點位於完全相同週期位置的 calibration scores；
- **BinaryLocal**：除了相同位置，也納入鄰近幾個週期位置；
- **ExpLocal**：所有 calibration samples 都保留，但依它們和目標週期位置的距離給予 exponential-decay weights。

BinaryPoint 的 local exchangeability 假設最強，資料規律時可得到很窄的區間；資料週期不穩定時容易失效。BinaryLocal 與 ExpLocal 假設較寬鬆，通常比較適合不規則的真實資料。

### 5.3 Remainder：globally exchangeable

如果 decomposition 足夠好，remainder 應接近不相關白噪音，因此作者對 remainder 主要採用 CV+。實驗中 CV+、EnbPI 與 ACI 在 remainder 上的表現相近，支持「良好分解後的 remainder 可近似 globally exchangeable」的想法。

但這個結論高度依賴 decomposition quality。若 trend 或 seasonality 沒有被正確抽出，remainder 仍含有結構性訊號，就不能合理視為 exchangeable。

## 6. 論文如何線性重組 component intervals

假設三個 component intervals 為：

\[
C_T=[L_T,U_T],\qquad
C_S=[L_S,U_S],\qquad
C_R=[L_R,U_R].
\]

作者採用 additive recomposition：

\[
L_Y=L_T+L_S+L_R,
\]

\[
U_Y=U_T+U_S+U_R,
\]

因此：

\[
C_Y=[L_T+L_S+L_R,\ U_T+U_S+U_R].
\]

這就是 component intervals 的 Minkowski sum。只要三個 component 同時落入各自區間，原始 (Y_t=T_t+S_t+R_t) 就一定落入重組區間。

作者也明確指出，這種線性重組經常 **overly conservative**。每個 component 都帶入自己的 uncertainty width，直接相加容易把 uncertainty 重複累積；它也沒有利用 component errors 之間的 covariance 或 error cancellation。

論文認為 recomposition model 和 decomposition model 同樣重要。如果真實成分之間比較接近 multiplicative interaction，直接相加上下界可能不適合。作者建議未來可調整各 component nominal coverage，或依 component magnitude、interval width 重新分配權重。

## 7. 為什麼每個 component 95%，final 不自動保證 95%

令 component miscoverage rates 為 (alpha_T,alpha_S,alpha_R)。若三者都設為 (alpha)，利用 union bound 可得到：

\[
P(Y_{n+1}\in C_Y)
\geq
1-(\alpha_T+\alpha_S+\alpha_R)
=1-3\alpha.
\]

所以三個 component 若各自是 95%：

\[
1-3(0.05)=0.85,
\]

只能得到很鬆的 85% lower bound，不代表實際 coverage 只有 85%。實際 coverage 經常更高，因為 union bound 很保守，而且 component errors 可能互相抵銷。

如果希望透過 Bonferroni 讓三個 component 的 simultaneous coverage 至少達到 95%，可令：

\[
\alpha_T=\alpha_S=\alpha_R=\frac{0.05}{3},
\]

也就是每個 component 約使用 98.33% interval。但作者沒有在實驗中採用這個修正，原因是直接線性重組本來就已經產生明顯 empirical overcoverage，再做 Bonferroni 會讓區間更寬。

對目前只有 low/high 兩個 components 的方法，對應結果是：

\[
P(Y_{n+1}\in C_Y)\geq1-(\alpha_L+\alpha_H).
\]

若 low 與 high 各自為 95%，一般 lower bound 是：

\[
1-0.05-0.05=0.90.
\]

若要 Bonferroni 保證 final 至少 95%，則應令：

\[
\alpha_L=\alpha_H=0.025,
\]

即 low/high 各自使用 97.5% interval。

### 對附錄 proof 的一個技術註記

附錄 A 把「sum 落入 summed interval」與「所有 components 同時各自被涵蓋」寫成等號。一般來說，較嚴謹的關係應是：

\[
\{T\in C_T,S\in C_S,R\in C_R\}
\subseteq
\{T+S+R\in C_T+C_S+C_R\},
\]

因為即使某個 component miss，errors 仍可能互相抵銷，使總和落入 summed interval。因此第一步應理解為大於等於，而不是必要的等號。不過 union-bound lower bound (1-\sum_j\alpha_j) 仍然成立。

## 8. 主要實驗結果

論文預設 nominal coverage 為 90%，評估：

- PICP：empirical interval coverage；
- PIAW：average interval width。

### 8.1 Synthetic data

Synthetic DGP 明確由線性 trend、正弦 seasonality 與 Gaussian remainder 相加，因此 decomposition structure 很清楚。

以 linear regressor 為例：

| 方法 | Coverage | Width |
|---|---:|---:|
| Raw signal EnbPI | 0.889 | 41.694 |
| 各成分都用 EnbPI 再線性重組 | 0.985 | 44.013 |
| Trend EnbPI + Seasonal BinaryPoint + Remainder CV+ | 0.946 | 29.755 |

結果顯示：

- 對所有成分都套 EnbPI 再相加，coverage 從約 89% 變成 98.5%，但更寬，明顯 overcoverage；
- BinaryPoint 能利用規律 seasonality，在仍高於 90% coverage 的情況下明顯縮小 width；
- decomposition 有效的前提是資料本身具有清楚、穩定且可恢復的 component structure。

### 8.2 San Diego energy consumption

Raw EnbPI 為 coverage 0.907、width 222.866；分解後方法多為 coverage 約 0.96–0.97、width 約 235–245。分解式方法多半 over-cover，但因能源資料週期較規律，BinaryPoint 對 seasonality 仍有一定效率。

### 8.3 Rossmann sales

這個資料具有多重且不穩定的 seasonality，但作者只使用能抽取單一季節性的 STL，因此 decomposition quality 不佳。Raw EnbPI 為 coverage 0.892、width 1572.343；分解後 linear-regression 方法甚至出現 coverage 1.0、width 約 5,000–6,000。

這是論文最重要的反例：分解不準會把誤差傳到所有 component CP procedures，最後線性相加後形成極寬、幾乎沒有實用性的區間。

### 8.4 Beijing air quality

Raw EnbPI 為 coverage 0.902、width 42.918；分解後常見 coverage 約 0.95–0.96、width 約 61–67。資料缺少穩定 seasonality，seasonal 與 remainder components 變得相似，顯示 decomposition 沒有成功辨認出明確結構。

## 9. 作者的主要結論與限制

作者認為這套方法最適合：

- component structure 明確；
- trend、seasonality、remainder 可被可靠分離；
- 對 component exchangeability 有合理先驗知識；
- 資料具有穩定週期或規律性。

主要限制包括：

1. **Decomposition quality 是上游瓶頸。** 分錯的訊號會傳入 predictor、nonconformity scores 與 recomposition interval。
2. **Linear recomposition 容易過度保守。** 它直接加總每個 component width，忽略 covariance 與 cancellation。
3. **各 component 的 coverage guarantees 不完全同質。** CV+、EnbPI、ACI、weighted CP 的 assumptions 與 finite/asymptotic guarantees 不同，很難得到乾淨統一的 final guarantee。
4. **Local exchangeability 依賴資料規律。** BinaryPoint 在規律 synthetic/energy data 有效，在複雜 sales/air-quality data 不穩定。
5. **沒有解決 optimal recomposition。** 如何分配 component miscoverage budgets、如何利用 covariance、如何處理 multiplicative structure，仍是 open problem。

## 10. 和目前 KF–ARIMA–ANN–EnbPI 的逐項比較

| 項目 | Prinzhorn et al. (2024) | 目前方法 |
|---|---|---|
| 原始結構 | (Y=T+S+R) | (Y=L+H)（模擬資料另含觀測雜訊，但 KF high 吸收剩餘） |
| 分解方法 | STL | causal local-level KF |
| 成分數量 | trend、seasonality、remainder 三流 | low、high 雙流 |
| Low/trend predictor | Linear/MLP/GB，主要比較 CP 方法 | ARIMA ensemble |
| High/season predictor | Linear/MLP/GB | MSE-ANN ensemble |
| Bootstrap/OOB | trend 可用 EnbPI | low/high 都使用 moving-block bootstrap 與 OOB/LOO EnbPI |
| Component interval | 每個成分各自 conformalize | low/high 各有各自 EnbPI residual pools |
| Final point | component point forecasts 相加 | ARIMA low + ANN high |
| Final interval | component lower 相加、upper 相加 | 對 combined OOB residual 重新做 EnbPI |
| 誤差相依性 | linear bounds sum 不直接估 covariance | combined residual pool 直接保留 low/high error 的共同結果 |
| Bias correction | 非本文重點 | 目前採 combined OOB mean bias correction |

因此，兩者的共同點是：

```text
分解 → 分流 predictor → component-wise uncertainty → additive reconstruction
```

但正式 final interval 不同：

### 論文版本

\[
C_Y^{\mathrm{sum}}
=
[L_L+L_H,\ U_L+U_H].
\]

### 目前版本

先建立 paired combined OOB residual：

\[
e_i^{Y}
=Y_i-(\widehat L_{-i}+\widehat H_{-i}),
\]

再直接取其 EnbPI offsets：

\[
C_Y^{\mathrm{direct}}
=
\widehat Y+[q_{\beta}(e^Y),q_{1-\alpha+\beta}(e^Y)].
\]

目前版本通常比 bounds 直接相加窄，因為 (e^Y=e^L+e^H) 的 empirical pool 已經保留兩個誤差的 correlation 與 cancellation。這不代表目前方法錯誤，而是採用了不同、通常更有效率的 final calibration target。

## 11. 這篇文獻是否比 EnbPI 原論文更接近目前研究

兩篇文獻負責不同部分：

- Xu and Xie 的 EnbPI 是目前 bootstrap/OOB residual calibration 與 sequential update 的方法來源；
- Prinzhorn et al. 是「將時間序列分解後，各流分別 conformalize，再重組 intervals」的直接方法來源。

所以目前研究可被描述成：

> 以 EnbPI 作為 component predictors 與 final predictor 的 calibration framework，再將 Prinzhorn et al. 的 component-wise conformal decomposition 概念改成 KF low/high 雙流，並比較 direct combined-residual calibration 與 component-bound linear recomposition。

這會比單純說「把 EnbPI 套到 KF 後」更能說明研究問題與方法差異。

## 12. 建議新增的正式比較實驗

不應直接覆蓋目前方法，而應在相同 data seeds、KF decomposition、ARIMA/ANN predictors 與 OOB predictions 下比較三種 final intervals：

### A. Direct combined-residual EnbPI（目前方法）

\[
C_Y^{A}=\widehat Y+C_{e^L+e^H}.
\]

它直接對最終預測目標 calibration，理論上最能利用 component error dependence。

### B. Component-wise 95% linear recomposition（Prinzhorn-style）

\[
C_Y^{B}=[L_L^{95}+L_H^{95},\ U_L^{95}+U_H^{95}].
\]

它最貼近本文 Eq. (4)，但 formal lower bound 只有 90%，實際上可能因 width 累加而 over-cover。

### C. Bonferroni component-wise recomposition

\[
C_Y^{C}=[L_L^{97.5}+L_H^{97.5},\ U_L^{97.5}+U_H^{97.5}].
\]

如果各 component guarantees 有效，這個版本可利用 union bound 取得至少 95% simultaneous-coverage lower bound，但預期最寬。

每個版本至少比較：

- final empirical coverage；
- mean/median interval width；
- interval score；
- point RMSE（應保持相同，否則不是純 interval comparison）；
- low/high OOB residual correlation；
- component simultaneous coverage；
- 因 error cancellation 而使 final covered、但 component 未同時 covered 的比例；
- 20 次 Monte Carlo paired differences 與標準差／confidence interval。

為了讓 B、C 真正對應論文，component bias correction 應在 low/high 各自完成後再重組。這和目前 combined-only bias correction 是另一個實驗因素，應固定或清楚分開，避免把 bias location 與 interval recomposition 同時改動。

## 13. 對目前研究最重要的啟示

1. 「component-wise conformal intervals 再相加」有正式文獻先例，因此可以成為合理 baseline。
2. 文獻本身沒有證明每個 component 都設 95% 就能讓 final 也是 95%；它只給出 (1-M\alpha) 的鬆下界，並承認 overall guarantee 仍是 open question。
3. 文獻實驗主要問題是 overcoverage 與區間過寬，而不是普遍 undercoverage。
4. 目前 direct combined-residual EnbPI 不是論文 Eq. (4)，但能自然利用 low/high error covariance，可能比線性 bounds sum 更窄。
5. 最有研究價值的問題不是「哪一種才算正確」，而是：

   > 在相同 KF decomposition 與 component predictors 下，direct conformalization of the summed forecast 和 linear recomposition of component-wise conformal intervals，何者能在維持 nominal coverage 時產生更有效率的區間？

6. 如果 linear recomposition 顯著 over-cover，而 direct combined EnbPI 接近 95% 且較窄，就能支持目前 final calibration 設計；如果 direct method under-cover，則應進一步檢查 residual dependence、distribution shift、component calibration validity 或採用 joint/multivariate conformal 方法。

## 14. 可對老師使用的簡短說法

> 我找到一篇和目前雙流方法很接近的文獻，Prinzhorn et al. (2024) 先把時間序列分成 trend、seasonality 和 remainder，對每個 component 分別做 conformal prediction，再把各 component 的 lower bounds 與 upper bounds 線性相加。作者指出，若每個 component 都使用 (1-\alpha) interval，重組後只能透過 union bound 得到 (1-M\alpha) 的鬆下界；若要用 Bonferroni 保證 final 95%，兩個 component 應各用 97.5%，但區間通常會過寬。這篇方法和我們的差別是，我們目前沒有直接相加 component bounds，而是對 low+high 的 combined OOB residual 再做一次 EnbPI，因此會保留兩個預測誤差的相關與抵銷效果。下一步可以把文獻的線性重組方法另做成 baseline，和目前 direct combined EnbPI 在相同 Monte Carlo paths 下比較 coverage、width 和 interval score。
