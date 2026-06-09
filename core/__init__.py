"""core/ - clean 3-class PPE detection engine (Person / Hardhat / Safety Vest).

Submodules:
    labels_tr   - whitelist + Turkish display strings (no heavy deps)
    postprocess - Layer 1: spatial detection cleanup (no heavy deps)
    violations  - Layers 2+3: association, smoothing, debounce/hysteresis (no heavy deps)
    detector    - v3 inference + ByteTrack + frame-skipping (imports torch/transformers)

`detector` is intentionally NOT imported here so that lightweight consumers
(and unit tests) can import labels_tr / postprocess / violations without torch.
"""
