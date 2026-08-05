# Universal visual location product contract

The product treats every reference as a **visual instance**. A reference may
be a still image, animated image, texture, UI element, person, rendered object,
effect, or sprite. Public `locate*` call signatures remain compatible with
PyAutoGUI 0.9.54.

## Runtime cascade

1. Predict a region from the previous two successful positions.
2. Route small references through canonical Rust matching and large references
   through TinyLocate CUDA inference.
3. Try the other fast path after a miss.
4. Search automatically generated scale variants as the exhaustive fallback.
5. Retain only high-confidence neural crops in a bounded in-memory appearance bank.
6. Track accepted candidates, expand the predicted region after misses, and then
   perform full-screen re-acquisition.
7. Fall back per operation when an accelerated backend reports an error.

The neural network is not an ONNX asset and is not embedded in the wheel.
Training exports a compact `TLN1` FP16 tensor stream. The runtime discovers an
external model from `PYAUTOGUI_TINYLOCATE_MODEL` or the local application model
directory and validates every tensor before allocation. CUDA inference is used
when its runtime is available; otherwise the existing Rust/PyScreeze matching
cascade remains active.

## Product gates

- Existing function signatures and aliases are unchanged.
- Missing models and unavailable GPUs preserve the established locate path.
- Still images, GIF representative frames, scale changes, movement, cache
  invalidation, corrupt models, and backend failures have side-effect-free tests.
- Accuracy reports include precision, recall, center error, IoU, false matches
  per hour, and time to re-acquire after occlusion.
- Performance reports include canonical hit, predicted-region hit, full-screen
  recovery, neural inference, and end-to-end p50/p95 latency.
- Wheels, sdists, editable installs, and Python 3.12/3.14 smoke tests are release
  gates. Neural training dependencies never become runtime dependencies.

## Current TinyLocate architecture

TinyLocate is a shared-weight reference/search encoder built from inverted
residual depthwise blocks, squeeze-excitation, a stride-8 normalized feature
map, reference-conditioned correlation, and objectness/box heads. The present
network has 359,281 learned parameters and exports to roughly 721 KiB of FP16
weights plus inference buffers.

The current training curriculum places up to six independently recolored,
rescaled, and flipped instances in each search, supervises every heatmap peak,
and combines per-instance box IoU with same-class hard-negative confidence loss.

The production extension includes precompiled CUDA compute kernels, while the
weights remain external. It links no PyTorch, ONNX, cuDNN, or TensorRT runtime.
On the RTX 3060 12 GB validation machine, the deterministic 256×256 fixture
measures 7.42 ms p50 and 7.82 ms p95 over 100 warm iterations. First-use model
validation, CUDA context creation, and parameter upload take about 93 ms.

The current 384×384 generated validation run reports 7.41 px mean center error,
93.13% center recall within 16 pixels, 99.58% within 32 pixels, and 0.609 mean
IoU. Six-instance generated searches reach 98.02% recall within 32 pixels and
99.63% within 64 pixels. A deterministic 500-pair Caltech101 transformed-image
run reports 14.09 px mean error and 78.80%/90.20%/96.20% recall within
16/32/64 pixels. A separate 1,000-pair exact-instance-negative run records a
5.8% crossing rate at the calibrated public 0.8 threshold; raw threshold 0.8
reduces that rate to 3.1% while retaining 91.8% score recall. These numbers are
release baselines rather than universal accuracy guarantees.

Large teacher networks may be used during training and distillation. Only the
compact student weight stream is distributed to runtime model storage.
