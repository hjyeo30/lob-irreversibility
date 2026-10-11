# Paper-to-code map

| Paper item | Calculation source | Contract or figure source |
|---|---|---|
| MBO reconstruction and source events | `reproduction/upstream/phases/02_book_reconstruction_features/codebook.py`, `reproduction/upstream/phases/03c_continuous_path_preflight/codebook.py`, `reproduction/src/block_producer.py` | `contracts/path_schema.md` |
| Physical reversal and 37-channel representation | `reproduction/upstream/phases/05_antisymmetric_sequence_models/codebook.py` | `contracts/reversal_registry.yaml`, Fig. 1 |
| Linear, pooled, and ordered estimators | `reproduction/upstream/phases/04_density_ratio_baselines/codebook.py`, `reproduction/upstream/phases/05_antisymmetric_sequence_models/codebook.py`, `reproduction/src/cuda_runtime.py` | linear values in Results; neural estimates throughout |
| Path length and event resolution | `reproduction/src/scale_runtime.py`, `reproduction/src/cuda_runtime.py` | Fig. 3 |
| Summed one-block reference | `reproduction/src/first_order_runtime.py` | Fig. 4(a) |
| Coarse transition surrogate | `reproduction/src/transition_surrogate_runtime.py` | `contracts/transition_surrogate_outputs.schema.json`, Fig. 4(a) |
| Session windows | `reproduction/src/regime_transfer_runtime.py` | `contracts/regimes.yaml` |
| Feature-family refits | `reproduction/src/ablation_runtime.py` | `contracts/feature_groups.yaml`, Fig. 4(b) |
| Twelve-symbol daytime analysis | `reproduction/src/cross_symbol_runtime.py`, `reproduction/src/cross_symbol_eda.py` | Fig. 2 |
| Return-prediction comparison | `reproduction/phases/10_predictive_bridge/codebook.py`, `reproduction/phases/11_lgbm_multihorizon_return/codebook.py`, `reproduction/phases/12_sigma_role_and_deoverlap/` | forecast statement in Results |
| Clock-free and padding controls | `reproduction/controls/short_term_controls.py`, `reproduction/src/cuda_runtime.py` | Fig. 5(b) |
| Shuffled block order | `reproduction/controls/shuffled_order_control.py` | Fig. 5(b) |
| Queue-reactive simulator | `reproduction/src/qr_real_input_adapter.py`, `reproduction/src/qr_simulator_runtime.py`, `reproduction/src/simulator_training_runtime.py` | Fig. 5(a--b) |
| Simulator transition surrogate | `reproduction/src/qr_transition_surrogate_runtime.py` | Fig. 5(c) |
| Figure 1 | `figures/fig_overview.html`, `figures/render_html.py` | vector PDF via Chromium |
| Figure 2 | `figures/make_figures.py`, `figures/data/cross_symbol.json` | mean rates and three test-date points per symbol |
| Figures 3--5 | `figures/make_figures.py` | PDF and high-resolution PNG |

The numerical arrays in the figure sources are the compact values reported in
the paper. They are retained so reviewers can reproduce the visual encodings
without access to licensed message-level data.
