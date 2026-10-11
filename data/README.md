# Data contract

No observations are included in this repository.

The analysis uses licensed Nasdaq ITCH market-by-order data from Databento
dataset `XNAS.ITCH`, schema `mbo`, for 15 trading days from 2026-07-02 through
2026-07-23. The main analysis uses NVDA during 10:00--15:30 New York time. The
cross-section additionally uses AAPL, COST, TSLA, SAIA, PEP, CROX, IWM, PODD,
SPY, QQQ, and SOFI.

Mount or copy user-supplied inputs below `/data` using this layout:

```text
/data/
├── lob_liquidity_time_irreversibility/
│   ├── raw/databento/xnas_itch/mbo/
│   │   └── symbol=NVDA/date=2026-07-02/xnas-itch-20260702.mbo.dbn.zst
│   └── derived/continuous_blocks_micro_v1/symbol=NVDA/
└── lob_irreversibility_interpretation_and_execution/
    └── derived/continuous_blocks_v1/symbol=SYMBOL/
```

The raw stream must retain exchange ordering and the integer price, timestamp,
order identifier, action, side, size, flags, and sequence fields supplied by
the vendor. Derived Parquet must satisfy `contracts/path_schema.md` and the
field/parity registry in `contracts/reversal_registry.yaml`.

Expected date split:

- development: 2026-07-02 through 2026-07-15 (9 trading days)
- calibration: 2026-07-16, 2026-07-17, 2026-07-20
- test: 2026-07-21, 2026-07-22, 2026-07-23

Do not fit scalers, coarse-state bins, feature selection, simulator parameters,
or calibration coefficients on test dates.
