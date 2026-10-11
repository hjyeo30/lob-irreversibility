# Artifact scope

The inclusion boundary is the submitted paper, not the full research workspace.

Included:

- source code that constructs an input used by a reported calculation
- source code that computes a reported result or a control named in the paper
- validation code required to establish a reported identity or support gate
- data contracts needed to interpret the omitted proprietary inputs
- exact publication-figure sources

Excluded:

- raw or derived market observations
- trained weights, predictions, checkpoints, and result tables
- private local account details, correspondence, and original research Git history
- manuscript drafts, response letters, internal plans, and exploratory reports
- paid acquisition helpers, provider credentials, SSH configuration, cloud
  scheduling, billing ledgers, and failure-recovery controllers
- experiments classified as out of scope for the submitted paper

Scientific validators are retained when they establish an identity or support
condition used in the paper. Provider-specific admission, dispatch, billing,
remote-copy, and recovery branches are not retained.

The public release includes paper authorship and citation metadata. Original code
and documentation are MIT-licensed; the bundled fonts retain their own license.
