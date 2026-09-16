# Changelog

## v1.3.0 / DAG-RandLoRA v1.0.0

- Added target-specific `block:q/v` basis ranks.
- Added exact fused-QKV full-weight gradient profiler.
- Added no-update generic calibration runner.
- Added gradient profile serialization.
- Added entropy effective-rank/GID statistics.
- Added R1 occupancy-conditioned block allocation.
- Added R2 Q/V-specific DAG allocation.
- Added fixed-budget dynamic-programming target allocator.
- Preserved v1.2 checkpoint/config behavior when `target_rank_pattern={}`.

## v1.2.0

See the retained v1.2 audit/validation history in the package for prior cache, merge, checkpoint, AMP and optimizer fixes.
