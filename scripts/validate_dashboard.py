from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    payload = json.loads((ROOT / "data" / "etf_meta_dashboard.json").read_text(encoding="utf-8"))
    quality = payload["metadata"]["quality"]
    expected = {
        "status": "pass",
        "requestedTickers": 12,
        "completeTickers": 12,
        "missingCells": 0,
        "duplicateDates": 0,
        "nonPositivePrices": 0,
    }
    for key, value in expected.items():
        if quality.get(key) != value:
            raise RuntimeError(f"品質檢查失敗：{key}={quality.get(key)!r}，預期 {value!r}")

    recommendation = payload["recommendation"]
    if recommendation["ticker"] not in {row["ticker"] for row in payload["sectors"]}:
        raise RuntimeError("建議ETF不在11類股清單中。")
    if len(payload["equity"]) < 100 or len(payload["gap"]) < 20:
        raise RuntimeError("圖表歷史資料筆數不足。")

    data_js = (ROOT / "assets" / "etf-meta-dashboard-data.js").read_text(encoding="utf-8")
    if not data_js.startswith("window.ETF_META_DATA="):
        raise RuntimeError("網頁資料檔格式錯誤。")
    index = (ROOT / "index.html").read_text(encoding="utf-8")
    if "RAY的投資筆記" not in index or "etf-meta-dashboard-data.js" not in index:
        raise RuntimeError("首頁缺少品牌署名或資料檔引用。")

    print(
        f"validated as_of={payload['metadata']['dataAsOf']} "
        f"mode={recommendation['mode']} ticker={recommendation['ticker']}"
    )


if __name__ == "__main__":
    main()
