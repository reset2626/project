# 學長論文重點整理

參考檔案：
- [學長論文.pdf](C:/Users/allen/OneDrive/Desktop/研究所/paper/學長論文.pdf)
- [senior_thesis_extracted.txt](C:/Users/allen/OneDrive/Desktop/程式/meeting/senior_thesis_extracted.txt)

## 方法主軸

學長的方法核心是把原始序列拆成低波動部分 `l_t` 和高波動部分 `h_t`，再分別處理：

- 低波動部分用 `ARIMA` 預測。
- 高波動部分用 `ANN` 預測。
- 最後總預測值是兩者相加，也就是 `\hat r_t = \hat l_t + \hat h_t`。

這個描述出現在第 3 章方法段，大意是把原本 Babu 的 `MA-filter` 換成 `Kalman filter`，並額外補上區間預測。

## 學長怎麼看「預測準不準」

### 1. 點預測

學長主要是用 `MSE` 來比較模型預測表現。

- 在太陽黑子資料的實證部分，文字直接寫到「MSE 的部分可以發現我們是比 Babu 的模型表現來得好」。
- 在元大台灣 50 的實證部分，也是同樣用 `MSE` 當主要比較標準。

也就是說，學長判斷點預測好不好，核心指標就是：

- `MSE`

在第 3 章 bootstrap 模擬實驗時，他也會拿「真實一步預測值」和各 bootstrap 方法的平均預測值相比，去看誰離真值比較近，但本質上仍是在看點預測誤差。

### 2. 區間預測

學長的區間預測不是用 coverage calibration 的角度在評估，而是用構造法先做出區間，再用圖和少量模擬例子去看。

他的作法是：

- `ARIMA` 本身提供低頻部分的區間。
- `ANN` 沒有天然區間，所以用 `bootstrap` 重複產生很多預測值。
- 將 bootstrap 預測值的 `2.5%` 與 `97.5%` 分位數拿來當高波動部分的區間。
- 最後把 `ARIMA` 的區間和 `bootstrap` 的區間相加，得到總體的 `95%` 預測區間。

論文第 3.3 節的大意就是：

- 低頻部分的 95% 區間由 ARIMA 殘差推得。
- 高頻部分的 95% 區間由 bootstrap 分位數給出。
- 最後把兩部分加總成完整區間。

## 區間「準不準」他怎麼看

這裡要特別注意，學長和你現在的看法不太一樣。

學長論文裡：

- 沒有看到像你現在這樣系統性地報 `coverage`。
- 也沒有看到用 `interval width` 或 `width-to-RMSE` 去檢查區間是不是太保守。
- 在真實資料部分，區間預測主要是畫圖展示。
- 在第 3 章 bootstrap 模擬例子裡，會看「真實值是否在信賴區間內」，但這比較像示範，不是大規模的 coverage 評估。

所以如果拿來對照你現在的結果，可以說：

- 學長當時重點主要放在 `MSE`。
- 區間部分比較偏「能不能做出一個合理區間」和「圖上看起來有沒有包住」。
- 沒有像你現在這樣把「區間是否過寬、是否只是用很大的不確定性包住」當成主要診斷問題。

## 和你現在狀況最有關的地方

論文後面其實有一段很值得注意：學長有試過「多濾波幾次」。

論文後段提到，他注意到一次 Kalman 濾波可能不夠，因此測試了多次濾波再分別讓 ARIMA 和 ANN 去擬合，但最後得到的 `MSE` 分別是：

- `373.3337`
- `389.5415`
- `403.1174`

結論是：

- 多濾波幾次並不會提升模型的預測能力。

這點和你現在看到的現象其實很接近：如果後面層數越做越像均值回歸，或區間越來越只是靠大不確定性包住，這和學長後面的觀察方向是一致的，也就是「多做幾層不一定比較好」。

## 你目前結果可以怎麼對照學長

如果照學長的評估邏輯：

- 你現在首先還是要看 `point MSE` 有沒有改善。
- 如果 `MSE` 沒改善，區間再漂亮也很難說方法比較好。

但如果照你現在更嚴格的角度：

- 還要額外看 `coverage`
- 還要看 `interval width`
- 還要看區間是不是只是被一個很大的不確定性撐開

所以你現在其實是比學長多做了一層更細的診斷：

- 學長比較重點在「點預測是否比 Babu 好」。
- 你現在則進一步檢查「區間是否只是保守亂包」。

## 簡短結論

可以很濃縮地說：

- 學長主要用 `MSE` 判斷點預測準不準。
- 區間預測是用 `ARIMA 區間 + bootstrap 分位數區間` 組合出來。
- 區間部分主要是畫圖展示和少量檢查真值是否落在區間內，沒有系統性地用 `coverage` 與 `width` 去檢驗是否過度保守。
- 論文後段也提到，多次濾波不一定會提升預測能力，這和你現在 iterative KF 的疑問很有關。

## Babu 論文怎麼評估

對照原始論文 `Babu and Reddy (2014), A moving-average filter based hybrid ARIMA–ANN model for forecasting time series data`，可以抓到幾個重點：

- 他們評估的是 `forecasting accuracy`，重心放在點預測。
- 原文摘要與 Results preview 明確寫到，他們同時看 `one-step-ahead` 和 `multistep-ahead` forecasts。
- 實驗資料包含：
  - `simulated data`
  - `sunspot data`
  - `electricity price data`
  - `stock market data`
- 比較對象包含：
  - 單獨 `ARIMA`
  - 單獨 `ANN`
  - 其他既有 hybrid ARIMA–ANN 方法

### Babu 的指標

從 ScienceDirect 的 article preview 可以直接確認：

- 他們在 Results and discussion 前，先定義了「兩個 performance measures」來比較 prediction accuracy。

由於 preview 內容把指標名稱截斷了，我無法只靠預覽頁逐字看到完整兩個名稱；但根據可檢索到的次級整理與引用摘要，這篇通常是以：

- `MAE`
- `MSE`

作為主要誤差衡量。

所以比較安全的說法是：

- **可以確定** Babu 的主軸是點預測誤差比較。
- **高度可能** 他主要使用的是 `MAE` 和 `MSE`。

### 和你現在方法的差別

Babu 的評估邏輯和你現在最在意的點不太一樣：

- Babu：重點是 `point forecast accuracy`
- 你現在：除了點預測，還會看
  - `coverage`
  - `interval width`
  - `width-to-RMSE`
  - 區間是不是只是被大不確定性包住

換句話說，Babu 那篇比較像是在回答：

- 「哪個模型點預測誤差比較小？」

而你現在其實在回答更進一步的問題：

- 「區間是不是有校準？」
- 「是不是只是用很大的 band 包住未來？」

### 對你目前結果的啟示

如果拿 Babu 的標準來看，你目前首先應該確認：

- `point MSE` 有沒有真的改善

但如果拿你現在更完整的標準來看，還要進一步問：

- 預測線是不是只是回到均值
- 區間是不是只是保守地包住
- 多做幾層 KF 之後，點預測有沒有真的變好

所以很簡單地說：

- **Babu 比較在意點預測誤差。**
- **你現在比 Babu 多做了區間品質診斷。**

## Babu 相關來源

- ScienceDirect article preview: https://www.sciencedirect.com/science/article/pii/S1568494614002555
- DOI: https://doi.org/10.1016/j.asoc.2014.05.028
