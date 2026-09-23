# Validation learning curves: raw reward and logarithmic gap

Campaign: kings-beta010-val128-n10; training seeds 10–19; 128 validation instances, four samples per instance; all seven methods, beta=0.1, 32 epochs.

The displayed transform is **S = -log10(1 - mean reward)**, applied to the per-epoch reward averaged over validation samples and training seeds. Reward is LB/makespan, with theoretical upper bound 1. The bound need not be attainable on every instance. We do not transform individual samples (some could equal 1), choose epsilon, fit a reference optimum, smooth curves, or use best-so-far envelopes.

An increase of 0.1 in S means the remaining gap is multiplied by 10^-0.1, or reduced by about 20.6%. This is a descriptive relative-gap scale, not evidence that discovery difficulty grows exponentially. The monotone transform preserves every per-epoch ranking of aggregate rewards; it cannot establish a row-delta advantage absent from the raw curve.

Bands are pointwise 95% t intervals over ten training seeds, transformed using the same monotone function. They do not include uncertainty from drawing a new validation set and are not simultaneous confidence bands. The paired-difference figure uses matched training seeds on the original scale.

The transform was chosen after inspecting results and is exploratory. No claims of statistical superiority or SOTA follow from this visualization. Final test results use validation-selected checkpoints and are different from final-epoch validation curves.
