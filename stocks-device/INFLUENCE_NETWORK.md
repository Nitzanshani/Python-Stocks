# Dynamic Influence Network (Phases 4–8 Design)

Influence means stable incremental predictive information, not economic
causality. A directed edge `A -> B` can exist only when source features improve
a target model outside the sample versus the same baseline without A.

The pipeline will first screen cheap lagged residual correlations, then run
expensive plugins only on candidates. Initial plugins are cross-correlation,
lagged regression, partial correlation, Granger/VAR and event response. Mutual
information and transfer entropy follow only after baseline validation.

Every result retains effect size, raw and Benjamini-Hochberg-adjusted p-values,
walk-forward improvement, stability and sample size. Network snapshots will be
stored by window; graph metrics and visualization remain a separate page and
are never presented as trading signals. DMD, Matrix Pencil, Prony and state-space
models remain a later research layer and will not modify Fourier `Freq Q`.

Phase 3A implements the small-universe directed comparison, not a dynamic
network. Its walk-forward split is purged and embargoed; preprocessing and
hyperparameter selection are fold-local. Event-study and Granger p-values use
separate declared FDR families. All Granger lags and both directions are kept.
The component scores and three weight sensitivities are experimental diagnostics
only; no production Overall Score has been adopted.
