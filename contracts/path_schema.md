---
contract: continuous_lob_path
version: 2
status: approved
storage_schema: continuous_blocks_v1
model_projection: continuous_path_37_v1
---

# Continuous LOB path schema

## 관측 단위

$$
W=(S_0,U_1,S_1,\ldots,U_K,S_K),\qquad
z_t=[S_{t-1},U_t,S_t].
$$

- \(S_t\): boundary state 7 channels
- \(U_t\): interval information 23 channels
- \(z_t\): block token 37 channels
- \(B\): block을 닫는 target source-event 수
- \(K\): 한 path의 block 수
- \(H=BK\): target event horizon

Block은 target count에 도달한 뒤 첫 valid completion-safe boundary에서 닫는다.
따라서 realized source-event count, completion overshoot와 elapsed time을
별도 보존한다. Path는 symbol, source date와 regime 경계를 넘지 않는다.

## Durable storage schema

새 block도 upstream 61-column `continuous_blocks_v1` Arrow schema를 그대로
사용한다. `regime`과 hard-reset 뒤의 `segment`는
`regime=<premarket|open|daytime|pre_close|after_hours>/segment=<NNNN>`
directory partition으로 표현하며 Parquet column을 추가하지 않는다.
`block_index`는 segment-local이다. 아래 field는 모두 non-null이다.

### Identity와 cursor

| Field | Arrow dtype | Unit/meaning |
|---|---|---|
| `dataset` | string | exact dataset ID |
| `symbol` | string | raw symbol |
| `source_date` | date32 | New York trading date |
| `target_b` | uint16 | target source-event count |
| `block_index` | uint32 | regime-segment-\(B\)-local contiguous index |
| `first_source_event_index` | uint64 | date-local source cursor |
| `last_source_event_index` | uint64 | date-local source cursor |
| `first_raw_ordinal` | uint64 | DBN record ordinal |
| `last_raw_ordinal` | uint64 | DBN record ordinal |
| `realized_source_event_count` | uint16 | actual block count |
| `completion_overshoot` | uint16 | actual count minus \(B\) |
| `block_start_ns` | int64 | UTC Unix ns |
| `block_end_ns` | int64 | UTC Unix ns |
| `elapsed_ns` | int64 | end minus start |
| `flags_or` | uint8 | bitwise OR of included raw flags |

### Start/end boundary state

각 prefix `start_`, `end_`에 다음 10 fields가 같은 dtype으로 존재한다.

| Suffix | Arrow dtype | Unit |
|---|---|---|
| `best_bid_nano` | int64 | \(10^{-9}\) currency unit |
| `best_ask_nano` | int64 | \(10^{-9}\) currency unit |
| `midprice_x2_nano` | int64 | bid + ask, \(10^{-9}\) unit |
| `spread_ticks` | float32 | validated ticks |
| `bid_depth_l1` | uint64 | displayed shares |
| `ask_depth_l1` | uint64 | displayed shares |
| `bid_depth_l5` | uint64 | displayed shares |
| `ask_depth_l5` | uint64 | displayed shares |
| `imbalance_l1` | float32 | \((B-A)/(B+A)\) |
| `imbalance_l5` | float32 | \((B-A)/(B+A)\) |

### Interval fields

| Field family | Exact fields | Arrow dtype |
|---|---|---|
| signed net depth | `bid_depth_l1_net_change`, `ask_depth_l1_net_change`, `bid_depth_l5_net_change`, `ask_depth_l5_net_change` | int64 |
| gross depth variation | same four prefixes with `_gross_variation` | uint64 |
| best depletion | `bid_best_depletion_count`, `ask_best_depletion_count` | uint32 |
| best appearance | `bid_best_appearance_count`, `ask_best_appearance_count` | uint32 |
| liquidity aggregates | `depletion_plus_appearance_count` uint32, `appearance_minus_depletion_count` int32 | as stated |
| return/quote movement | `mid_log_return` float64, `mid_change_half_ticks` int32, `bid_quote_change_ticks` int32, `ask_quote_change_ticks` int32 | as stated |
| absolute variation | `mid_abs_variation_half_ticks`, `spread_abs_variation_ticks` | uint64 |
| localization sidecar | `same_price_refill_count`, `cause_add_count`, `cause_cancel_count`, `cause_replace_count`, `cause_displayed_execution_count`, `cause_other_count` | uint32 |

Arrow schema metadata는 다음 exact key를 가진다.

```text
schema_version = continuous_blocks_v1
scientific_unit = completion-closed source-event-count block
price_unit = 1e-9 currency units
```

Clock partition 해석의 timezone은 `America/New_York`이지만 upstream과 exact
schema equality를 유지하기 위해 Arrow metadata key로 추가하지 않는다.

## Primary 37-channel projection

Durable 61 columns 전체가 model feature는 아니다. `continuous_path_37_v1`은
다음 exact order만 사용한다.

### Boundary 7

```text
spread_ticks
bid_depth_l1
ask_depth_l1
bid_depth_l5
ask_depth_l5
imbalance_l1
imbalance_l5
```

### Interval 23

```text
bid_depth_l1_net_change
ask_depth_l1_net_change
bid_depth_l5_net_change
ask_depth_l5_net_change
appearance_minus_depletion_count
mid_log_return
mid_change_half_ticks
bid_quote_change_ticks
ask_quote_change_ticks
bid_depth_l1_gross_variation
ask_depth_l1_gross_variation
bid_depth_l5_gross_variation
ask_depth_l5_gross_variation
depletion_plus_appearance_count
mid_abs_variation_half_ticks
spread_abs_variation_ticks
elapsed_ns
realized_source_event_count
completion_overshoot
bid_best_depletion_count
bid_best_appearance_count
ask_best_depletion_count
ask_best_appearance_count
```

한 token의 exact order는 `start_` boundary 7, interval 23, `end_` boundary
7이다.

## Transform와 scaling

- boundary의 spread와 four depth fields: `log1p`
- boundary imbalance: identity
- signed net, appearance-minus-depletion와 signed quote movement:
  `sign(x) * log1p(abs(x))`
- `mid_log_return`: identity
- nonnegative interval count/variation/time: `log1p`

Scaler는 development dates에서만 forward와 physical reverse orientation을
같이 사용해 channel-wise median과 IQR/1.349로 fit한다. IQR scale이 finite하지
않거나 \(10^{-8}\)보다 작으면 1을 사용한다. Boundary scaler는 start/end와
temporal position 전체에서 공유하고 interval scaler도 position 전체에서
공유한다.

## Storage-only/excluded field

Best Bid/Ask와 raw midpoint price, raw/source cursor, `flags_or`,
`same_price_refill_count`, 다섯 cause counts, `dataset`, `symbol`,
`source_date`, directory의 `regime`·`segment`, \(B\)와 block index는
provenance·QA·localization을
위해 storage에 남긴다. Primary 37-channel tensor에는 들어가지 않는다.
Absolute timestamp, raw price level, order ID와 raw cause mark는 primary model
feature로 사용하지 않는다.

## Upstream adapter

Upstream `continuous_blocks_v1`과 `continuous_blocks_micro_v1`은 동일한
61-column schema다. Adapter는 physical file을 변형하지 않고 partition
metadata로 다음을 주입한다.

```text
regime = daytime
storage_schema = continuous_blocks_v1
model_projection = continuous_path_37_v1
```

Adapter는 upstream path와 `regime=daytime`만 읽는다. Upstream Parquet를
rewrite하거나 새 fingerprint를 만들지 않는다.
