# Paper experiment matrix

This file fixes the paper-facing experiment universe independently of the
machine-specific scheduler used for the original runs. Provider provisioning,
leases, billing ledgers, and remote recovery controllers are excluded.

All neural fits use the development dates declared by `src.cuda_runtime`, the
three held-out dates declared there, seeds `20260730`, `20260731`, and
`20260732`, development-only scaling, a nonnegative scalar temperature fitted
on calibration dates, and date-equal test aggregation. Unless a row says
otherwise, the source is NVDA, `B=10`, the representation has 37 channels, and
the score is clipped to `[-log(99), log(99)]` before division by realized source
events.

| Result | Retained calculation source | Exact unit universe or retained boundary |
|---|---|---|
| path scale | `src.scale_runtime.run_scale_unit` | `K in {1,2,3,4,6,8,12}` x pooled/ordered x 3 seeds at `B=10`; report clipped and unclipped length rates |
| event resolution | `src.scale_runtime.run_scale_unit` | `(B,K) in {(5,12),(10,6),(30,2)}` x pooled/ordered x 3 seeds, plus `(10,2)` for the matched-`K` contrast |
| one-block sum | `src.first_order_runtime.run_first_order_unit` | `K in {2,6,12}` x 3 seeds |
| coarse surrogate | `src.transition_surrogate_runtime` construction and scoring implementations | 20 surrogate randomizations x 3 held-out dates; ordered models at `K in {2,6,12}` x 3 seeds |
| session windows | `src.regime_transfer_runtime` | separately fitted `edge` and `daytime` units x pooled/ordered x 3 seeds; open and pre-close are edge-component evaluations |
| feature ablation | `src.ablation_runtime.run_ablation_unit` | four feature groups, no-book-state, and full representation x pooled/ordered x 3 seeds |
| cross-section | `src.cross_symbol_runtime` | 12 paper symbols x pooled/ordered x 3 seeds, daytime only |
| clock-free | `controls.short_term_controls` | pooled/ordered x 3 seeds, `K=6`, 34 channels; transform/model definitions retained, original machine driver omitted |
| circular padding | `controls.short_term_controls` | ordered x 3 seeds, `K=6`, 37 channels; model definition retained, original machine driver omitted |
| shuffled order | `controls.shuffled_order_control` | ordered x 3 seeds, `K=6`, 34 channels; pooled anchor reused only after the exact equality gate; transform/gate retained, original machine driver omitted |
| QR benchmark | `src.qr_simulator_runtime`, `src.simulator_training_runtime` | real anchors and QR-A/QR-B 37-channel variants x pooled/ordered x 3 seeds; QR-A/QR-B share an exact 34-channel input after the identity gate; generation, unit universe, QA, and aggregation retained, model-launch adapter omitted |
| QR surrogate | `src.qr_transition_surrogate_runtime` | 5 surrogate repetitions x 3 model seeds, evaluated date-equally over 3 held-out dates |

The exact input schemas are in `contracts/`. Each retained runtime takes an
explicit input and output root; no default path authorizes data acquisition.
Aggregation must reject missing units and average the three held-out dates with
equal weight before summarizing across seeds. Figure rendering consumes only the
compact published values embedded in `figures/`.
