# TimeSeries Project 補充稿

這份是依照作業 PDF 重整後的 `Problem 2` 與 `Problem 3`。

重點是：

- `Problem 2` 已不是單純 `constant mean + GARCH`，而是要先配適 **ARIMA mean model**。
- `Problem 2` 的 forecast origin 要固定在 **`2008-12-31`**。
- `Problem 3` 是新的 **Seasonal ARIMA model**，你原本 notebook 還沒做。

---

# Problem 2：GARCH 模型

## 題目和你原本 notebook 的差別

你原本的 notebook 主要是這兩點要改：

1. 題目要求先對 `Y_t` 建立一個合適的 **ARIMA mean model**。
2. 題目要求在 **`2008-12-31`** 做 1-step 到 5-step ahead forecast。

所以新版寫法應該是：

- 先把月報酬率轉成 `log-return`。
- 再用 `ARIMA` 配適平均方程。
- 對 `ARIMA` 殘差做 ARCH effect 檢定。
- 再對殘差建立 `GARCH(1,1)` 模型。
- `Y_t` 的預測由 ARIMA mean model 提供。
- volatility 的預測由 GARCH variance model 提供。

## 1. 資料前處理與 log-return 計算

```python
m3_df = pd.read_csv(
    r"c:\Users\allen\OneDrive\Desktop\研究所\時序\Dataset_3M.txt",
    sep=r"\s+"
)

m3_df.columns = m3_df.columns.str.strip()
m3_df["date"] = pd.to_datetime(m3_df["date"].astype(str)).dt.to_period("M").dt.to_timestamp("M")
m3_df["rtn"] = pd.to_numeric(m3_df["rtn"], errors="coerce")
m3_df["Y"] = np.log1p(m3_df["rtn"])
m3_df = m3_df.dropna().set_index("date")

# 題目指定預測起點為 2008/12/31
forecast_origin = pd.Timestamp("2008-12-31")
Y_train = m3_df.loc[:forecast_origin, "Y"]

plt.figure(figsize=(10, 4))
plt.plot(Y_train)
plt.title("3M 股票 log-return（截至 2008-12-31）")
plt.xlabel("Date")
plt.ylabel("Y_t")
plt.grid(True)
plt.show()
```

### 可直接寫的中文說明

本題使用 `Dataset_3M.txt` 中 3M 股票的月報酬率資料，並依題目要求將其轉換為對數報酬率：

\[
Y_t = \log \left(\frac{P_t}{P_{t-1}}\right)
\]

由於題目指定 forecast origin 為 `2008-12-31`，因此後續的模型建立與預測皆以前述日期之前的資料作為樣本。

---

## (a) ARCH effect 檢定

```python
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

candidate_orders = [(0,0,1), (1,0,0), (1,0,1), (2,0,0), (2,0,1), (2,0,2), (3,0,0)]
mean_results = []

for order in candidate_orders:
    try:
        fit = ARIMA(Y_train, order=order).fit()
        lb_p = acorr_ljungbox(fit.resid.dropna(), lags=[10], return_df=True)["lb_pvalue"].iloc[0]
        mean_results.append({
            "order": order,
            "AIC": fit.aic,
            "BIC": fit.bic,
            "LB(10) p-value": lb_p
        })
    except Exception:
        pass

mean_table = pd.DataFrame(mean_results).sort_values("AIC")
display(mean_table)

mean_model = ARIMA(Y_train, order=(2, 0, 2))
mean_fit = mean_model.fit()
print(mean_fit.summary())

mean_resid = mean_fit.resid.dropna()

arch_test = het_arch(mean_resid - mean_resid.mean(), nlags=10)
print("ARCH LM statistic:", arch_test[0])
print("ARCH LM p-value:", arch_test[1])
print("F statistic:", arch_test[2])
print("F p-value:", arch_test[3])
```

### 可直接寫的中文說明

為了檢驗報酬率序列是否具有 ARCH effect，先依題目要求對 `Y_t` 建立 mean model。比較多個 ARIMA 候選模型後，可選擇 `ARIMA(2,0,2)` 作為平均方程，因其 AIC 表現較佳，且殘差 Ljung-Box 檢定結果亦合理。

接著對 `ARIMA(2,0,2)` 的殘差進行 Engle 的 ARCH LM 檢定。

檢定假設如下：

- 虛無假設 `H0`：殘差不存在 ARCH effect。
- 對立假設 `H1`：殘差存在 ARCH effect。

由檢定結果可見，p-value 很小，小於 0.05，因此拒絕虛無假設，表示殘差具有顯著的 ARCH effect，適合進一步建立 GARCH 模型。

---

## (b) 建立 GARCH 模型

```python
from arch import arch_model

garch_model = arch_model(mean_resid, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
garch_fit = garch_model.fit(disp="off")
print(garch_fit.summary())
```

### 可直接寫的中文說明

在確認 mean equation 的殘差存在 ARCH effect 後，可對殘差建立 `GARCH(1,1)` 模型。由於平均部分已由 `ARIMA(2,0,2)` 捕捉，因此在 GARCH 階段採用 `mean="Zero"`，只針對條件變異數建模。

本題所建立的模型可表示為：

\[
Y_t = \mu_t + \varepsilon_t
\]

其中 \(\mu_t\) 由 `ARIMA(2,0,2)` 提供，而殘差項滿足：

\[
\varepsilon_t = \sigma_t z_t,\quad z_t \sim N(0,1)
\]

\[
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
\]

此模型可用來描述金融報酬率常見的波動群聚現象。

---

## (c) GARCH 模型殘差診斷

```python
std_resid = pd.Series(garch_fit.std_resid, index=mean_resid.index).dropna()

plt.figure(figsize=(10, 4))
plt.plot(std_resid)
plt.title("標準化殘差")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(std_resid**2)
plt.title("標準化殘差平方")
plt.grid(True)
plt.show()

print(acorr_ljungbox(std_resid, lags=[10], return_df=True))
print(acorr_ljungbox(std_resid**2, lags=[10], return_df=True))
```

### 可直接寫的中文說明

為檢查所建立的 GARCH 模型是否合理，可觀察標準化殘差與標準化殘差平方，並進一步進行 Ljung-Box 檢定。

- 若標準化殘差的 p-value 不小，表示殘差不具有顯著自相關。
- 若標準化殘差平方的 p-value 不小，表示模型已有效捕捉原本的 ARCH effect。

若兩者檢定結果皆合理，便可認為此 GARCH 模型的配適效果可接受。

---

## (d) 波動預測

```python
variance_forecast = garch_fit.forecast(horizon=5).variance.iloc[-1]
volatility_forecast = np.sqrt(variance_forecast)

pd.DataFrame({
    "variance": variance_forecast.values,
    "volatility": volatility_forecast.values
}, index=[f"{i}-step" for i in range(1, 6)])
```

### 可直接寫的中文說明

根據已建立的 `GARCH(1,1)` 模型，可在 forecast origin `2008-12-31` 下進行 1-step 至 5-step ahead 的條件變異數與波動預測。

若預測結果隨步數增加而逐漸趨於平穩，表示模型呈現典型的波動均值回歸特性，這也是 GARCH 模型常見的現象。

---

## (e) 報酬率預測

```python
mean_forecast = mean_fit.get_forecast(steps=5).summary_frame()[["mean", "mean_ci_lower", "mean_ci_upper"]]
mean_forecast
```

### 可直接寫的中文說明

本題中的 `Y_t` 預測應由 **ARIMA mean model** 提供，而不是直接用 constant mean。因為題目已明確要求先建立適當的 ARIMA 模型作為平均方程。

因此，在 forecast origin `2008-12-31` 下，可利用 `ARIMA(2,0,2)` 對未來 1-step 到 5-step 的 `Y_t` 進行預測，並同時報告其 confidence interval。

---

# Problem 3：Seasonal ARIMA 模型

## 1. 載入資料與初步觀察

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

hsales_df = pd.read_csv(r"c:\Users\allen\OneDrive\Desktop\研究所\時序\hsales.csv")
hsales = hsales_df.iloc[:, -1].astype(float)
hsales.index = pd.date_range("1973-01-01", periods=len(hsales), freq="MS")

plt.figure(figsize=(10, 4))
plt.plot(hsales)
plt.title("USA 新建獨棟住宅月銷售量")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.show()
```

### 可直接寫的中文說明

`hsales.csv` 為美國新建獨棟住宅的月銷售量資料。由時間序列圖可觀察到資料具有明顯的季節波動，且不同時期的波動幅度不完全一致，因此適合進一步考慮轉換與季節模型。

---

## (a) 資料是否需要轉換

```python
log_hsales = np.log(hsales)

plt.figure(figsize=(10, 4))
plt.plot(log_hsales)
plt.title("log(hsales)")
plt.xlabel("Date")
plt.ylabel("log sales")
plt.grid(True)
plt.show()
```

### 可直接寫的中文說明

由於原始資料的變異程度會隨水準改變，因此先對資料取對數轉換較為合適。對數轉換可以穩定變異數，並使季節與趨勢型態更容易分析，因此本題採用 `log(hsales)` 作為後續建模基礎。

---

## (b) 平穩性檢查與差分

```python
print("ADF p-value of raw series:", adfuller(hsales)[1])
print("ADF p-value of log series:", adfuller(log_hsales)[1])
print("ADF p-value after seasonal differencing:", adfuller(log_hsales.diff(12).dropna())[1])

seasonal_diff = log_hsales.diff(12).dropna()

plt.figure(figsize=(10, 4))
plt.plot(seasonal_diff)
plt.title("log(hsales) seasonal differencing (lag = 12)")
plt.xlabel("Date")
plt.ylabel("Differenced log sales")
plt.grid(True)
plt.show()
```

### 可直接寫的中文說明

為檢查資料是否平穩，可對原始序列、對數轉換後序列，以及季節差分後序列進行 ADF 檢定。

由結果可知，直接觀察原始序列與 log 序列時仍可看出季節成分，因此再進行 `lag = 12` 的季節差分。季節差分後的序列較接近平穩，因此後續 Seasonal ARIMA 模型可採用 seasonal differencing。

---

## (c) 模型辨識

```python
sarima_candidates = [
    ((1,0,0), (0,1,1,12)),
    ((2,0,0), (0,1,1,12)),
    ((1,0,1), (0,1,1,12)),
    ((2,0,1), (0,1,1,12)),
    ((1,0,0), (1,1,1,12))
]

sarima_results = []
for order, seasonal_order in sarima_candidates:
    fit = SARIMAX(
        log_hsales,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True
    ).fit(disp=False)
    resid = fit.resid.dropna().iloc[24:]
    lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
    sarima_results.append({
        "order": order,
        "seasonal_order": seasonal_order,
        "AIC": fit.aic,
        "BIC": fit.bic,
        "LB(12) p-value": lb["lb_pvalue"].iloc[0],
        "LB(24) p-value": lb["lb_pvalue"].iloc[1]
    })

pd.DataFrame(sarima_results).sort_values("AIC")
```

### 可直接寫的中文說明

在模型辨識階段，可比較數個合理的 Seasonal ARIMA 候選模型，並以 AIC、BIC 與殘差檢定結果作為選模依據。

綜合比較後，可選擇：

\[
\text{SARIMA}(1,0,0)\times(0,1,1)_{12}
\]

作為本題的候選模型，因其在配適品質與殘差診斷方面皆有不錯表現。

---

## (d) 參數估計與殘差檢定

```python
sarima_fit = SARIMAX(
    log_hsales,
    order=(1, 0, 0),
    seasonal_order=(0, 1, 1, 12),
    trend="n",
    enforce_stationarity=True,
    enforce_invertibility=True
).fit(disp=False)

print(sarima_fit.summary())

sarima_resid = sarima_fit.resid.dropna().iloc[24:]
print(acorr_ljungbox(sarima_resid, lags=[12, 24], return_df=True))
```

### 可直接寫的中文說明

在估計 `SARIMA(1,0,0) × (0,1,1,12)` 後，可檢查模型參數與殘差表現。

再利用 Ljung-Box 檢定確認殘差是否仍存在顯著自相關。若 p-value 不小，則表示殘差可視為近似白噪音，代表模型已大致捕捉資料中的動態結構與季節性。

---

## (e) 未來 24 個月預測

```python
forecast_24 = sarima_fit.get_forecast(steps=24).summary_frame()
forecast_24_level = np.exp(forecast_24[["mean", "mean_ci_lower", "mean_ci_upper"]])
forecast_24_level.index = pd.date_range(log_hsales.index[-1] + pd.offsets.MonthBegin(), periods=24, freq="MS")

forecast_24_level
```

```python
plt.figure(figsize=(12, 5))
plt.plot(hsales, label="Observed")
plt.plot(forecast_24_level.index, forecast_24_level["mean"], label="Forecast", color="red")
plt.fill_between(
    forecast_24_level.index,
    forecast_24_level["mean_ci_lower"],
    forecast_24_level["mean_ci_upper"],
    color="pink",
    alpha=0.3,
    label="95% CI"
)
plt.title("24 個月預測結果")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
```

### 可直接寫的中文說明

最後利用選定的 Seasonal ARIMA 模型對未來 24 個月進行預測。由於模型是在 log 尺度上估計，因此預測結果需取指數後再轉回原始銷售量尺度，並同時呈現 confidence interval。

此結果可用來描述未來兩年的住宅銷售量可能走勢，以及其不確定性範圍。
