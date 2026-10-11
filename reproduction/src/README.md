# Scientific runtime index

These modules are grouped below by the paper question they answer. Keeping the
shared estimator and validation functions together avoids duplicating model
semantics across thin command wrappers.

## Core path and estimator layer

| File | Scientific responsibility |
|---|---|
| `block_producer.py` | Multi-regime continuous-block publication and parity checks |
| `cuda_runtime.py` | Deterministic neural fitting, calibration, scoring, and resume parity |
| `scale_runtime.py` | Path-length and event-resolution units |

## Identifying comparisons

| File | Scientific responsibility |
|---|---|
| `first_order_runtime.py` | Summed one-block reference |
| `transition_surrogate_runtime.py` | Coarse-transition-preserving surrogate construction and scoring |
| `regime_transfer_runtime.py` | Edge/daytime fits and open/pre-close evaluation |
| `ablation_runtime.py` | Feature-family masks and refits |
| `cross_symbol_runtime.py` | Twelve-symbol model-unit definitions and validation |
| `cross_symbol_eda.py` | Compact cross-symbol descriptives and correlations |

## Queue-reactive benchmark

| File | Scientific responsibility |
|---|---|
| `qr_real_input_adapter.py` | Ordered MBO replay into QR calibration and clock inputs |
| `qr_simulator_runtime.py` | Calibration, deterministic simulation, Arrow publication, and QA |
| `simulator_training_runtime.py` | Neural-unit universe, 34-channel identity gate, and aggregation contracts |
| `qr_transition_surrogate_runtime.py` | QR-A transition surrogate and date-equal scoring |

Production-only provisioning, remote transfer, billing, and recovery code was
removed. Some modules therefore expose calculation functions without a
standalone launcher; the precise boundary is stated in
[`../drivers/README.md`](../drivers/README.md), not hidden behind placeholder
commands.
