# 84日相對報酬 Meta 策略儀表板

以 252 日超額 SPY 排名建立 Top1 與 Top2 產業輪動策略，再用最近 84 個交易日的相對績效與 ±1% 持有帶選擇當期模式。

## 線上版本

https://mp781237.github.io/etf-meta-84d-dashboard/

## 自動更新

GitHub Actions 每週六 12:00（Asia/Taipei）直接在 GitHub 雲端下載最新完整交易日資料、重算策略、通過品質檢查後部署 GitHub Pages。更新不依賴任何本機電腦。

也可以在儲存庫的 Actions 頁面手動執行 `Update ETF data and deploy Pages`。

## 網頁內容

- 當月建議持股與完整決策鏈
- Meta、Top1、Top2、固定 50/50 與 SPY 累積資產比較
- Gap84 與歷史模式切換
- 11 大類股多期間表現與狀態
- 策略規則、價格來源、資料品質與風險提醒

資料來源為 Yahoo Finance 調整後日線；ETF 類股分類以 State Street 官方資料核對。歷史回測與頁面內容不構成投資建議。
