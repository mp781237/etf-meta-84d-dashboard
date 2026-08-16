from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from update_dashboard import (  # noqa: E402
    TICKERS,
    align_panels_to_complete_sessions,
    calculate_meta,
    merge_ticker_panels,
    weekly_signal_dates,
)


class DashboardStrategyTests(unittest.TestCase):
    def test_all_price_panels_drop_a_session_missing_one_ticker(self) -> None:
        dates = pd.bdate_range("2026-08-10", periods=3)
        panels = {
            field: pd.DataFrame(100.0, index=dates, columns=TICKERS)
            for field in ("Open", "High", "Low", "Close", "Volume")
        }
        panels["Close"].at[dates[1], "XLE"] = float("nan")

        aligned, incomplete = align_panels_to_complete_sessions(panels)

        self.assertEqual(incomplete.tolist(), [dates[1]])
        for frame in aligned.values():
            self.assertEqual(frame.index.tolist(), [dates[0], dates[2]])

    def test_unused_high_low_gap_does_not_drop_a_strategy_session(self) -> None:
        dates = pd.bdate_range("2026-08-10", periods=3)
        panels = {
            field: pd.DataFrame(100.0, index=dates, columns=TICKERS)
            for field in ("Open", "High", "Low", "Close", "Volume")
        }
        panels["High"].at[dates[1], "XLE"] = float("nan")

        aligned, incomplete = align_panels_to_complete_sessions(panels)

        self.assertTrue(incomplete.empty)
        for frame in aligned.values():
            self.assertEqual(frame.index.tolist(), dates.tolist())

    def test_individual_ticker_download_repairs_a_missing_cell(self) -> None:
        dates = pd.bdate_range("2026-08-10", periods=2)
        panels = {
            field: pd.DataFrame(100.0, index=dates, columns=TICKERS)
            for field in ("Open", "High", "Low", "Close", "Volume")
        }
        panels["Close"].at[dates[1], "XLE"] = float("nan")
        replacement = {
            field: pd.DataFrame({"XLE": [101.0, 102.0]}, index=dates)
            for field in panels
        }

        merge_ticker_panels(panels, replacement, "XLE")

        self.assertEqual(panels["Close"].at[dates[1], "XLE"], 102.0)

    def test_weekly_signal_uses_last_available_session(self) -> None:
        dates = pd.DatetimeIndex(["2026-07-27", "2026-07-29", "2026-07-31", "2026-08-03"])
        self.assertEqual(
            weekly_signal_dates(dates),
            {pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-03")},
        )

    def test_meta_switches_only_after_weekly_decision(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=100)
        top1 = pd.Series(100_000.0, index=dates)
        top2 = pd.Series(100_000.0, index=dates)
        top1.iloc[84:] = 102_000.0
        _, gap, decision, _ = calculate_meta(top1, top2)
        first_qualified = gap[gap > 0.01].index[0]
        first_weekly = min(date for date in weekly_signal_dates(dates) if date >= first_qualified)
        self.assertEqual(decision.loc[first_weekly], 1.0)
        if first_weekly > first_qualified:
            self.assertEqual(decision.loc[first_qualified], 0.5)


if __name__ == "__main__":
    unittest.main()
