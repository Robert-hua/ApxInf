# Kart DINOv2 diffusion policy

`kart_dp` is a separate native family for the custom Kart model. It does not
reuse the LeRobot ResNet Diffusion topology or change any existing family.
The first profile is batch 1, four historical frames, state width 62,
DINOv2 ViT-S/14 (384 channels, 12 blocks, six attention heads), compact
FiLM temporal U-Net, and ten DDIM steps with eta=0. It returns the entire
24-by-3 chunk in steering/throttle/brake order. Canonical ranges are
[-1,1], [0,1], [0,1]; physical units and robot execution are outside this port.

The maintained CUDA path is in `crates/apxinf-model/src/kart_dp/`:
`config` validates the supported profile; `weights` loads the safetensors map;
`model` owns DINO, pooling, conditioning, U-Net and DDIM math;
`model_runner` owns buffers, eager execution, graph capture and tensor binding.
`policies/impls/kart_dp.py` owns observation preprocessing and action clipping.
The native path has no Torch/TensorRT runtime; an explicit optional TensorRT
vision path is documented below.
The forward temporal convolutions explicitly select prepared FP32 im2col/GEMM;
vision and transposed convolutions retain cuDNN. This family-local choice needs
no process-wide provider environment override and does not change other models.
DINO LayerNorm and attention Softmax explicitly use `layer_norm_block` and
`softmax_block` (operation kinds 13/14, `block_reductions.cuh`). These independent
kernels retain synchronized broadcast/reduction reuse; legacy kinds 4/5 and
`primitives.cuh` are restored to their pre-Kart-fix implementation. The old
multi-warp race remains a documented legacy limitation, not a claimed fix for
other families. GroupNorm and other unchanged primitives retain existing paths.
Offline export/reference tools remain in the parent ApexForge evaluation directory and require the original
training source. A training `.pt` must first be exported; ordinary and EMA
weights are different candidates and the choice is explicit in the export.

Public loading uses `AutoPolicy.from_pretrained(path, model_type="kart_dp",
model_variant="f32")`. Native detection uses `config.json:model_type`.
`f32` remains default. Experimental `fp16` selects half operands for Linear
and temporal forward/transposed Conv1d; accumulation, output, nonlinear math,
DDIM, vision convolutions and attention remain FP32. `fp8_policy` changes policy
Linears to per-tensor E4M3 and otherwise follows `fp16`; it is not an all-FP8
model. BF16, other shapes, samplers and dynamic batch sizes are unsupported.
Candidate admission does not imply numerical acceptance. `prepare(observation, mode="graph")` captures one full graph;
`mode="eager"` clears it. `infer` accepts:

- `observation.images.front`: RGB uint8 `[4,240,320,3]` or already center-cropped
  `[4,224,288,3]`, oldest first. The larger image uses crop y=8, x=16.
- `observation.state`: raw finite float `[4,62]`, in training state order.
- optional `noise`: finite float `[1,24,3]`; by default uses checkpoint fixed noise.

The caller owns temporal selection according to checkpoint obs_stride=4 and
must not substitute four duplicate current frames. Frame resizing to 320x240
belongs to the application and must match the training OpenCV INTER_AREA path.
Policy applies checkpoint state statistics, three-pixel left/right edge padding,
and ImageNet normalization. The native input seam is canonical FP32
`[4,3,224,294]`, state `[1,248]`, and explicit noise `[1,24,3]`.
There are no hidden history queues. `actions` is clipped and `prediction` is
unclipped DDIM output. Each returned array is independently owned.

Original position/CLS/normalization/noise/frequency constants remain in the
export for strict original-source reference loading. Their policy or prepared
replacements are explicit. The unused mask token is excluded only because
unmasked inference is the supported profile. Export records hashes of the
checkpoint, source, files, selected weight branch and static transforms.
A registry entry or successful build does not establish numerical acceptance;
see the parent ApexForge PROJECT_STATUS for current hardware evidence and limits.

## Explicit TensorRT vision candidate

The Python policy accepts `vision_engine=...` as an opt-in hybrid backend.
The engine manifest must match the engine and checkpoint SHA, carry
`input_contract=kart_dp_vision_features_v1`, and expose FP32 images
`[1,4,3,224,288]` to features `[1,64,384]`. TensorRT owns image padding,
normalization and DINO/projection/pooling. The native runner consumes flat
features `[1,24576]`, state `[1,248]`, noise `[1,24,3]` and runs conditioning
plus DDIM10. The named `vision_features` manifest explicitly enables that seam;
ordinary native input is unchanged. Original vision assets are accounted as
externally owned, not silently discarded as unused keys.

The initial hybrid handoff copies features through host memory (TRT D2H then
native H2D). It uses independent streams with synchronization at the handoff;
`prepare(mode="graph")` captures separate TRT vision and native policy graphs,
while `mode="eager"` clears both. Vision capture retains its engine, context,
stream and stable device addresses; changing image values updates the input
before every launch. The
complete public call includes both engines and transfers. Do not describe this
as all-native, zero-copy or a single whole-model graph. TensorRT/CUDA Python
bindings are optional; loading normal `kart_dp` has no TensorRT dependency.
Parent ApexForge evaluation tools build engines on the target host and report
actual layer precision, FP32-reference action error and complete-call latency.

## High-accuracy native development variants

`f32_gemm` replaces only temporal transposed convolutions with prepared strict
F32 GEMM/gather. `f32_compensated` additionally selects independently prepared
FP16 high/low three-product Linear and temporal Conv1d providers (FP32 output,
accumulation and bias); one-row Linears remain strict F32. These are compensated
approximations, not bitwise FP32 and not ordinary FP16 inference.

`f32_fast` adds packed F32 fused attention from pinned CUTLASS example 41
(TF32 three-product arithmetic, head dim 64), independent warp LayerNorm,
LayerNorm-to-Linear splitting, and scaled residual kernels. Exact-erf GELU and
the fc2 input hi/lo split share a kernel with the original FP32 rounding order.
It prepares immutable timestep embeddings once and
computes all ten per-step FiLM projections as batches once per new observation.
FiLM inputs still depend on the current observation. Fusion (state/vision
projection) executes once, followed by all ten DDIM U-Net/sampling iterations.
DDPM training does not make this inference loop DDPM.

With native vision, `f32_fast` uploads four RGB uint8 frames each call and
normalizes/edge-pads on the GPU through an independently owned input buffer.
The FP32 input seam remains available. No historical image/feature cache is
used: all four frames and the entire vision graph execute every request.
The original `f32` default and existing families retain their provider choices.

With explicit `autotune=True`, preparation additionally searches cuBLASLt
tactics for the compensated products, retaining FP32 accumulation/output and
the existing bias epilogue. Tuning executes once before either eager or Graph
mode; capture never changes the chosen buffers. Preparation time increases,
and this is an opt-in development candidate rather than a new default precision.
The parent `custom-kernels` report records the source/runtime and acceptance.

Target-device 64-window development checks and replay pass for the current
candidate; this is not full validation, a 10 ms guarantee, or robot acceptance.
The parent report `reports/thor/kart-dp-epoch200/native-fast/README.md` records
exact binaries, timing boundaries, rejected variants and pending acceptance.
