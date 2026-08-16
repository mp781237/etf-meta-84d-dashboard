from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA_JS = ASSETS / "etf-meta-dashboard-data.js"
DATA_JSON = ROOT / "data" / "etf_meta_dashboard.json"
PRICE_CACHE = ROOT / "data" / "etf_price_history.csv.gz"

DOWNLOAD_START = "2018-06-18"
TEST_START = "2019-07-01"
REQUIRED_PRICE_FIELDS = ("Open", "Close")
LOOKBACK_DAYS = 252
META_LOOKBACK_DAYS = 84
META_BAND = 0.01
COST_BPS_PER_SIDE = 5

BENCHMARK = "SPY"
UNIVERSE = [
    "XLC",
    "XLY",
    "XLP",
    "XLE",
    "XLF",
    "XLV",
    "XLI",
    "XLB",
    "XLRE",
    "XLK",
    "XLU",
]
TICKERS = [BENCHMARK, *UNIVERSE]

SECTORS = {
    "XLC": "通訊服務",
    "XLY": "非必需消費",
    "XLP": "必需消費",
    "XLE": "能源",
    "XLF": "金融",
    "XLV": "醫療保健",
    "XLI": "工業",
    "XLB": "原物料",
    "XLRE": "房地產",
    "XLK": "科技",
    "XLU": "公用事業",
}

OFFICIAL_URLS = {
    ticker: (
        "https://www.ssga.com/us/en/intermediary/etfs/"
        f"state-street-{slug}-select-sector-spdr-etf-{ticker.lower()}"
    )
    for ticker, slug in {
        "XLC": "communication-services",
        "XLY": "consumer-discretionary",
        "XLP": "consumer-staples",
        "XLE": "energy",
        "XLF": "financial",
        "XLV": "health-care",
        "XLI": "industrial",
        "XLB": "materials",
        "XLRE": "real-estate",
        "XLK": "technology",
        "XLU": "utilities",
    }.items()
}


def _extract_download(raw: pd.DataFrame, requested: list[str]) -> dict[str, pd.DataFrame]:
    panels = {
        field: pd.DataFrame(index=raw.index)
        for field in ("Open", "High", "Low", "Close", "Volume")
    }
    if raw.empty:
        return panels
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        for field in panels:
            if field in level0:
                frame = raw[field].copy()
                if isinstance(frame, pd.Series):
                    frame = frame.to_frame(requested[0])
                panels[field] = frame
    elif len(requested) == 1:
        for field in panels:
            if field in raw:
                panels[field] = raw[[field]].rename(columns={field: requested[0]})
    for field, frame in panels.items():
        frame.columns = [str(column).strip().upper().replace(".", "-") for column in frame.columns]
        panels[field] = frame
    return panels


def weekly_signal_dates(dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
    grouped = pd.Series(dates, index=dates).groupby(dates.to_period("W-FRI"))
    return set(pd.to_datetime(grouped.last().tolist()))


def build_excess_spy_hold_band_targets(
    close: pd.DataFrame,
    keep_rank: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = close[[*UNIVERSE, BENCHMARK]].pct_change(LOOKBACK_DAYS, fill_method=None)
    excess = returns[UNIVERSE].sub(returns[BENCHMARK], axis=0)
    score = excess.copy()
    score[BENCHMARK] = 0.0
    targets = pd.DataFrame(False, index=close.index, columns=[*UNIVERSE, BENCHMARK])
    start_location = close.index.searchsorted(pd.Timestamp(TEST_START))
    if start_location <= 0:
        raise ValueError("測試起點前必須至少有一個交易日。")
    first_signal = close.index[start_location - 1]
    current: str | None = None
    for signal_date in sorted(weekly_signal_dates(close.index)):
        if signal_date < first_signal:
            continue
        row = score.loc[signal_date, UNIVERSE].dropna()
        if row.empty:
            current = BENCHMARK
        else:
            ranks = row.rank(ascending=False, method="min")
            leader = str(row.idxmax())
            if float(row.loc[leader]) <= 0:
                current = BENCHMARK
            elif current in UNIVERSE and float(ranks.loc[current]) <= keep_rank:
                pass
            else:
                current = leader
        targets.at[signal_date, current] = True
    return targets, score


def rotation_backtest(
    panels: dict[str, pd.DataFrame],
    target_mask: pd.DataFrame,
    score: pd.DataFrame,
    test_end: str,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = panels["Close"]
    open_ = panels["Open"]
    all_dates = close.index
    dates = all_dates[
        (all_dates >= pd.Timestamp(TEST_START)) & (all_dates <= pd.Timestamp(test_end))
    ]
    signal_dates = weekly_signal_dates(all_dates)
    cost_rate = COST_BPS_PER_SIDE / 10_000
    cash = 100_000.0
    positions: dict[str, int] = {}
    transactions: list[dict] = []
    equity_rows: list[dict] = []

    for date in dates:
        location = all_dates.get_loc(date)
        previous_date = all_dates[location - 1] if location > 0 else None
        if previous_date in signal_dates:
            available = target_mask.loc[previous_date].fillna(False)
            targets = available.index[available].tolist()
            targets = sorted(
                targets,
                key=lambda ticker: (
                    -float(score.at[previous_date, ticker])
                    if pd.notna(score.at[previous_date, ticker])
                    else math.inf
                ),
            )[:1]
            equity_open = cash + sum(
                shares * float(open_.at[date, ticker])
                for ticker, shares in positions.items()
                if np.isfinite(open_.at[date, ticker])
            )
            desired: dict[str, int] = {}
            for ticker in targets:
                price = open_.at[date, ticker]
                if np.isfinite(price) and price > 0:
                    desired[ticker] = math.floor(equity_open / float(price))

            for ticker in list(positions):
                sell_shares = max(positions[ticker] - desired.get(ticker, 0), 0)
                if sell_shares <= 0:
                    continue
                price = float(open_.at[date, ticker])
                gross = sell_shares * price
                cost = gross * cost_rate
                cash += gross - cost
                positions[ticker] -= sell_shares
                if positions[ticker] == 0:
                    del positions[ticker]
                transactions.append({"side": "sell", "cost": cost, "strategy": label})

            for ticker in targets:
                current_shares = positions.get(ticker, 0)
                buy_shares = max(desired.get(ticker, 0) - current_shares, 0)
                if buy_shares <= 0:
                    continue
                price = float(open_.at[date, ticker])
                affordable = math.floor(cash / (price * (1 + cost_rate)))
                buy_shares = min(buy_shares, affordable)
                if buy_shares <= 0:
                    continue
                gross = buy_shares * price
                cost = gross * cost_rate
                cash -= gross + cost
                positions[ticker] = current_shares + buy_shares
                transactions.append({"side": "buy", "cost": cost, "strategy": label})

        invested = sum(
            shares * float(close.at[date, ticker])
            for ticker, shares in positions.items()
            if np.isfinite(close.at[date, ticker])
        )
        equity_rows.append({"date": date.date().isoformat(), "equity": cash + invested})

    if dates.empty:
        raise RuntimeError("回測期間沒有可用交易日。")
    final_date = dates[-1]
    for ticker, shares in list(positions.items()):
        gross = shares * float(close.at[final_date, ticker])
        cost = gross * cost_rate
        cash += gross - cost
        transactions.append({"side": "final_sell", "cost": cost, "strategy": label})
    equity_rows[-1]["equity"] = cash
    return pd.DataFrame(equity_rows), pd.DataFrame(transactions)


def iso_date(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def finite_or_none(value: float) -> float | None:
    return round(float(value), 8) if math.isfinite(float(value)) else None


def align_panels_to_complete_sessions(
    panels: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    complete = pd.Series(True, index=panels["Close"].index)
    for field in REQUIRED_PRICE_FIELDS:
        complete &= panels[field][TICKERS].notna().all(axis=1)
    complete_index = panels["Close"].index[complete]
    if complete_index.empty:
        raise RuntimeError("12檔ETF沒有共同完整Open/Close交易日。")
    incomplete_index = panels["Close"].index[~complete]
    aligned = {
        field: frame.reindex(index=complete_index, columns=TICKERS).copy()
        for field, frame in panels.items()
    }
    aligned["Volume"] = aligned["Volume"].fillna(0)
    return aligned, incomplete_index


def merge_ticker_panels(
    panels: dict[str, pd.DataFrame],
    replacement: dict[str, pd.DataFrame],
    ticker: str,
) -> None:
    for field, frame in panels.items():
        if ticker not in replacement[field].columns:
            continue
        patch = replacement[field][ticker].reindex(frame.index)
        frame[ticker] = frame[ticker].combine_first(patch)


def load_price_cache(max_date: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    cache = {field: pd.DataFrame() for field in REQUIRED_PRICE_FIELDS}
    if not PRICE_CACHE.exists():
        return cache
    flat = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True, compression="gzip")
    flat.index = pd.DatetimeIndex(flat.index).tz_localize(None)
    if max_date is not None:
        flat = flat.loc[flat.index <= max_date.normalize()]
    for field in REQUIRED_PRICE_FIELDS:
        prefix = f"{field}__"
        columns = [column for column in flat.columns if column.startswith(prefix)]
        frame = flat[columns].copy()
        frame.columns = [column.removeprefix(prefix) for column in columns]
        cache[field] = frame.reindex(columns=TICKERS)
    return cache


def merge_price_cache(
    panels: dict[str, pd.DataFrame],
    cache: dict[str, pd.DataFrame],
) -> None:
    cached_dates = cache["Close"].index
    if cached_dates.empty:
        return
    all_dates = panels["Close"].index.union(cached_dates).sort_values()
    for field in panels:
        panels[field] = panels[field].reindex(index=all_dates, columns=TICKERS)
    for field in REQUIRED_PRICE_FIELDS:
        cached = cache[field].reindex(index=all_dates, columns=TICKERS)
        panels[field] = panels[field].combine_first(cached).reindex(columns=TICKERS)


def save_price_cache(panels: dict[str, pd.DataFrame]) -> None:
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    flat = pd.concat(
        [panels[field].add_prefix(f"{field}__") for field in REQUIRED_PRICE_FIELDS],
        axis=1,
    )
    flat.to_csv(
        PRICE_CACHE,
        date_format="%Y-%m-%d",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


def repair_incomplete_tickers(
    panels: dict[str, pd.DataFrame],
    exclusive_end: str,
) -> None:
    affected = [
        ticker
        for ticker in TICKERS
        if any(panels[field][ticker].isna().any() for field in REQUIRED_PRICE_FIELDS)
    ]
    for ticker in affected:
        raw = yf.download(
            ticker,
            start=DOWNLOAD_START,
            end=exclusive_end,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
            timeout=30,
        )
        replacement = _extract_download(raw, [ticker])
        merge_ticker_panels(panels, replacement, ticker)


def download_prices(end: str | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    end_date = pd.Timestamp(end) if end else pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None)
    exclusive_end = (end_date.normalize() + pd.Timedelta(days=2)).date().isoformat()
    panels: dict[str, pd.DataFrame] | None = None
    for attempt in range(1, 4):
        try:
            raw = yf.download(
                TICKERS,
                start=DOWNLOAD_START,
                end=exclusive_end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
                timeout=30,
            )
            candidate = _extract_download(raw, TICKERS)
            for field in candidate:
                candidate[field] = candidate[field].reindex(columns=TICKERS).sort_index()
            recent_dates = candidate["Close"].index[-10:]
            recent_incomplete = pd.Series(False, index=recent_dates)
            for field in REQUIRED_PRICE_FIELDS:
                recent_incomplete |= candidate[field].loc[recent_dates, TICKERS].isna().any(axis=1)
            panels = candidate
            if not recent_incomplete.any() or attempt == 3:
                break
            print(
                f"Yahoo近期資料不完整，第{attempt}次下載後重試："
                f"{[iso_date(date) for date in recent_dates[recent_incomplete]]}"
            )
        except Exception:
            if attempt == 3:
                raise
        time.sleep(attempt * 3)
    if panels is None:
        raise RuntimeError("Yahoo價格下載失敗。")
    repair_incomplete_tickers(panels, exclusive_end)
    cache = load_price_cache(end_date if end else None)
    merge_price_cache(panels, cache)

    close = panels["Close"]
    valid = [ticker for ticker in TICKERS if not close[ticker].dropna().empty]
    missing_tickers = sorted(set(TICKERS) - set(valid))
    common = close.dropna(how="any")
    if missing_tickers or common.empty:
        raise RuntimeError(f"價格下載不完整：{missing_tickers or '沒有共同交易日'}")
    latest_by_ticker = {ticker: close[ticker].last_valid_index() for ticker in TICKERS}
    common_last = common.index[-1]
    stale_tickers = [
        ticker for ticker, date in latest_by_ticker.items() if date != common_last
    ]
    panels, incomplete_dates = align_panels_to_complete_sessions(panels)
    incomplete_in_test = incomplete_dates[incomplete_dates >= pd.Timestamp(TEST_START)]
    if not incomplete_in_test.empty:
        raise RuntimeError(
            "Yahoo個別重抓後仍有不完整交易日："
            f"{[iso_date(date) for date in incomplete_in_test]}"
        )
    common = panels["Close"]
    common_last = common.index[-1]
    non_positive = int((common <= 0).sum().sum())
    duplicate_dates = int(common.index.duplicated().sum())
    if stale_tickers or non_positive or duplicate_dates:
        raise RuntimeError(
            "價格品質檢查失敗："
            f"stale={stale_tickers}, non_positive={non_positive}, duplicates={duplicate_dates}"
        )

    save_price_cache(panels)

    metadata = {
        "source": "Yahoo Finance through yfinance",
        "priceType": "auto-adjusted daily Open/Close",
        "firstDate": iso_date(common.index[0]),
        "lastDate": iso_date(common_last),
        "sessions": int(len(common)),
        "requestedTickers": len(TICKERS),
        "completeTickers": len(valid),
        "missingCells": int(common.isna().sum().sum()),
        "droppedIncompleteSessions": int(len(incomplete_dates)),
        "cachedSessions": int(len(cache["Close"])),
        "duplicateDates": duplicate_dates,
        "nonPositivePrices": non_positive,
        "status": "pass",
    }
    return panels, metadata


def undo_final_liquidation(
    equity: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.Series:
    series = pd.Series(
        equity["equity"].to_numpy(dtype=float),
        index=pd.to_datetime(equity["date"]),
    )
    if not transactions.empty:
        final_cost = transactions.loc[
            transactions["side"].eq("final_sell"), "cost"
        ].sum()
        series.iloc[-1] += float(final_cost)
    return series


def run_shadow_strategy(
    panels: dict[str, pd.DataFrame],
    test_end: str,
    keep_rank: int,
    label: str,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    targets, score = build_excess_spy_hold_band_targets(panels["Close"], keep_rank)
    equity, transactions = rotation_backtest(panels, targets, score, test_end, label)
    return undo_final_liquidation(equity, transactions), targets, score


def calculate_meta(
    top1: pd.Series,
    top2: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    top1_return = top1.pct_change(fill_method=None)
    top2_return = top2.pct_change(fill_method=None)
    top1_return.iloc[0] = top1.iloc[0] / 100_000 - 1
    top2_return.iloc[0] = top2.iloc[0] / 100_000 - 1
    gap = (top1 / top1.shift(META_LOOKBACK_DAYS)) / (
        top2 / top2.shift(META_LOOKBACK_DAYS)
    ) - 1
    periods = pd.Series(gap.index.to_period("W-FRI"), index=gap.index)
    decision_day = periods.ne(periods.shift(-1))

    state = 0.5
    decisions: list[float] = []
    for date, value in gap.items():
        if decision_day.loc[date] and pd.notna(value):
            if value > META_BAND:
                state = 1.0
            elif value < -META_BAND:
                state = 0.0
        decisions.append(state)

    decision = pd.Series(decisions, index=gap.index)
    allocation = decision.shift(1).fillna(0.5)
    gross_return = allocation * top1_return + (1 - allocation) * top2_return
    turnover = allocation.diff().abs().fillna(0)
    net_return = (1 + gross_return) * (1 - turnover * 0.001) - 1
    equity = (1 + net_return).cumprod() * 100_000
    return equity, gap, decision, turnover


def metrics(equity: pd.Series) -> dict:
    returns = equity.pct_change(fill_method=None).dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    drawdown = equity / equity.cummax() - 1
    annual = (1 + returns).groupby(returns.index.year).prod() - 1
    return {
        "ending": round(float(equity.iloc[-1]), 2),
        "totalReturn": finite_or_none(equity.iloc[-1] / 100_000 - 1),
        "cagr": finite_or_none((equity.iloc[-1] / 100_000) ** (1 / years) - 1),
        "volatility": finite_or_none(returns.std() * np.sqrt(252)),
        "sharpe": finite_or_none(returns.mean() / returns.std() * np.sqrt(252)),
        "maximumDrawdown": finite_or_none(drawdown.min()),
        "return2025": finite_or_none(annual.get(2025, np.nan)),
        "return2026": finite_or_none(annual.get(2026, np.nan)),
    }


def target_on(targets: pd.DataFrame, signal_date: pd.Timestamp) -> str:
    selected = targets.columns[targets.loc[signal_date].fillna(False)].tolist()
    return str(selected[0]) if selected else BENCHMARK


def effective_date(index: pd.DatetimeIndex, signal_date: pd.Timestamp) -> pd.Timestamp:
    location = index.searchsorted(signal_date, side="right")
    if location < len(index):
        return index[location]
    return signal_date + pd.offsets.BDay(1)


def build_holding_changes(
    signal_dates: list[pd.Timestamp],
    common_index: pd.DatetimeIndex,
    decision: pd.Series,
    top1_targets: pd.DataFrame,
    top2_targets: pd.DataFrame,
) -> list[dict]:
    changes: list[dict] = []
    previous: dict[str, str] | None = None
    for signal_date in signal_dates:
        if signal_date not in decision.index:
            continue
        top1_holding = target_on(top1_targets, signal_date)
        top2_holding = target_on(top2_targets, signal_date)
        mode = "Top1" if decision.loc[signal_date] == 1 else "Top2"
        active_holding = top1_holding if mode == "Top1" else top2_holding
        current = {
            "mode": mode,
            "activeHolding": active_holding,
            "top1Holding": top1_holding,
            "top2Holding": top2_holding,
        }
        if previous and current != previous:
            changes.append(
                {
                    "signalDate": iso_date(signal_date),
                    "effectiveDate": iso_date(effective_date(common_index, signal_date)),
                    "mode": mode,
                    "activeHolding": active_holding,
                    "previousActiveHolding": previous["activeHolding"],
                    "top1Holding": top1_holding,
                    "top2Holding": top2_holding,
                    "previousTop1Holding": previous["top1Holding"],
                    "previousTop2Holding": previous["top2Holding"],
                    "modeChanged": mode != previous["mode"],
                }
            )
        previous = current
    return changes


def classify_sector(rank: int, excess_252: float, excess_21: float, excess_63: float) -> str:
    if rank <= 2 and excess_252 > 0:
        return "領先"
    if rank <= 4 and excess_252 > 0:
        return "強勢"
    if excess_21 > 0 and excess_63 > 0:
        return "改善"
    if excess_21 > 0:
        return "反彈"
    return "落後"


def sector_snapshot(close: pd.DataFrame, latest: pd.Timestamp) -> list[dict]:
    returns = {days: close.pct_change(days, fill_method=None) for days in [5, 21, 63, 252]}
    excess = {
        days: returns[days][UNIVERSE].sub(returns[days][BENCHMARK], axis=0)
        for days in returns
    }
    score = excess[252].loc[latest]
    ranks = score.rank(ascending=False, method="min").astype(int)
    month_start_location = close.index.searchsorted(latest.replace(day=1))
    prior_month_close = close.iloc[max(month_start_location - 1, 0)]
    month_return = close.loc[latest] / prior_month_close - 1
    spy_month_return = float(month_return[BENCHMARK])

    rows = []
    for ticker in UNIVERSE:
        rank = int(ranks[ticker])
        row = {
            "ticker": ticker,
            "sector": SECTORS[ticker],
            "price": round(float(close.at[latest, ticker]), 2),
            "week": finite_or_none(returns[5].at[latest, ticker]),
            "month": finite_or_none(month_return[ticker]),
            "quarter": finite_or_none(returns[63].at[latest, ticker]),
            "year": finite_or_none(returns[252].at[latest, ticker]),
            "excessWeek": finite_or_none(excess[5].at[latest, ticker]),
            "excessMonth": finite_or_none(month_return[ticker] - spy_month_return),
            "excessQuarter": finite_or_none(excess[63].at[latest, ticker]),
            "excessYear": finite_or_none(excess[252].at[latest, ticker]),
            "rank": rank,
            "status": classify_sector(
                rank,
                float(excess[252].at[latest, ticker]),
                float(excess[21].at[latest, ticker]),
                float(excess[63].at[latest, ticker]),
            ),
            "officialUrl": OFFICIAL_URLS[ticker],
            "historyUrl": f"https://finance.yahoo.com/quote/{ticker}/history/",
        }
        rows.append(row)
    return sorted(rows, key=lambda item: item["rank"])


def sampled_equity(series: dict[str, pd.Series]) -> list[dict]:
    frame = pd.concat(series, axis=1).dropna(how="any")
    weekly = frame.groupby(frame.index.to_period("W-FRI")).tail(1)
    if weekly.index[-1] != frame.index[-1]:
        weekly = pd.concat([weekly, frame.tail(1)]).sort_index()
    rows = []
    for date, values in weekly.iterrows():
        row = {"date": iso_date(date)}
        row.update({key: round(float(value), 2) for key, value in values.items()})
        rows.append(row)
    return rows


def build_payload(end: str | None = None) -> dict:
    panels, quality = download_prices(end)
    common_close = panels["Close"].dropna(how="any")
    latest = common_close.index[-1]
    test_end = iso_date(latest)
    top1, top1_targets, _ = run_shadow_strategy(panels, test_end, 1, "Top1")
    top2, top2_targets, _ = run_shadow_strategy(panels, test_end, 2, "Top2")
    meta, gap, decision, turnover = calculate_meta(top1, top2)
    fixed = 0.5 * top1 + 0.5 * top2
    spy_close = common_close.loc[top1.index, BENCHMARK]
    spy = 100_000 * spy_close / spy_close.iloc[0]

    signal_dates = sorted(weekly_signal_dates(common_close.index))
    last_signal = signal_dates[-1]
    top1_target = target_on(top1_targets, last_signal)
    top2_target = target_on(top2_targets, last_signal)
    meta_mode = "Top1" if decision.loc[last_signal] == 1 else "Top2"
    recommendation = top1_target if meta_mode == "Top1" else top2_target
    execute_on = effective_date(common_close.index, last_signal)
    holding_changes = build_holding_changes(
        signal_dates, common_close.index, decision, top1_targets, top2_targets
    )

    changes = decision[decision.ne(decision.shift()) & decision.isin([0.0, 1.0])]
    switch_rows = []
    for signal_date, state in changes.items():
        mode = "Top1" if state == 1 else "Top2"
        targets = top1_targets if mode == "Top1" else top2_targets
        switch_rows.append(
            {
                "signalDate": iso_date(signal_date),
                "effectiveDate": iso_date(effective_date(common_close.index, signal_date)),
                "mode": mode,
                "holding": target_on(targets, signal_date),
                "gap84": finite_or_none(gap.loc[signal_date]),
            }
        )

    latest_top1_84 = top1.iloc[-1] / top1.iloc[-1 - META_LOOKBACK_DAYS] - 1
    latest_top2_84 = top2.iloc[-1] / top2.iloc[-1 - META_LOOKBACK_DAYS] - 1
    sector_rows = sector_snapshot(common_close, latest)
    recommended_sector = SECTORS.get(recommendation, "S&P 500")
    recommended_price = float(common_close.at[latest, recommendation])
    generated_at = pd.Timestamp.now(tz=ZoneInfo("Asia/Taipei"))

    payload = {
        "metadata": {
            "generatedAt": generated_at.isoformat(timespec="seconds"),
            "dataAsOf": test_end,
            "signalDate": iso_date(last_signal),
            "executeOn": iso_date(execute_on),
            "recommendationMonth": execute_on.strftime("%Y-%m"),
            "testStart": TEST_START,
            "lookbackDays": LOOKBACK_DAYS,
            "metaLookbackDays": META_LOOKBACK_DAYS,
            "metaBand": META_BAND,
            "costBpsPerSide": COST_BPS_PER_SIDE,
            "quality": quality,
        },
        "recommendation": {
            "ticker": recommendation,
            "sector": recommended_sector,
            "referencePrice": round(recommended_price, 2),
            "priceDate": test_end,
            "mode": meta_mode,
            "top1Target": top1_target,
            "top2Target": top2_target,
            "gap84": finite_or_none(gap.loc[last_signal]),
            "top1Return84": finite_or_none(latest_top1_84),
            "top2Return84": finite_or_none(latest_top2_84),
            "officialUrl": OFFICIAL_URLS.get(recommendation),
            "historyUrl": f"https://finance.yahoo.com/quote/{recommendation}/history/",
        },
        "metrics": {
            "meta": metrics(meta),
            "top1": metrics(top1),
            "top2": metrics(top2),
            "equal": metrics(fixed),
            "spy": metrics(spy),
        },
        "equity": sampled_equity(
            {"meta": meta, "top1": top1, "top2": top2, "equal": fixed, "spy": spy}
        ),
        "gap": [
            {
                "date": iso_date(date),
                "value": finite_or_none(gap.loc[date]),
                "decision": (
                    "Top1" if decision.loc[date] == 1 else "Top2" if decision.loc[date] == 0 else "50/50"
                ),
            }
            for date in gap.index
            if date in weekly_signal_dates(gap.index) and pd.notna(gap.loc[date])
        ],
        "sectors": sector_rows,
        "switches": switch_rows,
        "holdingChanges": holding_changes,
        "sources": [
            {
                "name": "Yahoo Finance via yfinance",
                "role": "策略計算主資料",
                "detail": "調整後日線開盤價與收盤價；延續原回測資料定義，避免來源切換造成訊號跳動。",
                "url": "https://finance.yahoo.com/",
            },
            {
                "name": "State Street Sector ETFs",
                "role": "ETF身分與類股定義",
                "detail": "官方基金頁用於確認11檔Select Sector SPDR ETF及其類股分類。",
                "url": "https://www.ssga.com/us/en/intermediary/capabilities/equities/sector-investing/sector-and-industry-etfs",
            },
            {
                "name": "GitHub雲端品質閘門",
                "role": "交易前檢查",
                "detail": "12檔日期必須一致、共同資料不得缺漏、價格必須為正且不得有重複日期。",
                "url": None,
            },
        ],
    }
    payload["metadata"]["switchCount"] = int((turnover > 0).sum())
    return payload


def save_payload(payload: dict) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    DATA_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    DATA_JS.write_text(f"window.ETF_META_DATA={encoded};\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="更新84日Top1／Top2 Meta策略網頁資料")
    parser.add_argument("--end", help="資料截止日，格式YYYY-MM-DD；預設下載至目前最新交易日")
    args = parser.parse_args()
    payload = build_payload(args.end)
    save_payload(payload)
    recommendation = payload["recommendation"]
    metadata = payload["metadata"]
    print(
        f"as_of={metadata['dataAsOf']} execute={metadata['executeOn']} "
        f"mode={recommendation['mode']} ticker={recommendation['ticker']} "
        f"gap84={recommendation['gap84']:.4f} price={recommendation['referencePrice']:.2f}"
    )


if __name__ == "__main__":
    main()
