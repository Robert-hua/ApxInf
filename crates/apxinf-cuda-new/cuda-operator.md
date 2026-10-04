# CUDA L3 Operator Catalog

This document lists the model-independent L3 semantics currently exposed by `apxinf-cuda-new`. Model code must match the complete mathematical semantic and tensor contract, not only the operator name. If no interface matches completely, record an operator gap and follow [`doc/adding-new-kernels.md`](../../doc/adding-new-kernels.md).

This document describes only public contracts and does not promise a specific provider, candidate, or autotune winner. Every operator supports eager execution and `prepare_with_session` → capture → replay.

## Shared Constraints

- Input, output, bias, and scale tensors must reside on the CUDA device of the current `CudaContext`.
- Tensors use contiguous row-major layout; output storage must not overlap read-only inputs.
- Shapes must be nonempty, and dimensions passed to the native layer must not exceed `i32::MAX`.
- Policy fields such as workspace, graph-safe, and deterministic affect only candidate eligibility and the recipe; they do not change the L3 mathematical semantic.
- `l3-operator` comments are machine-readable markers. A unit test compares them with the Rust semantic metadata registered by each operator family to ensure that every public semantic appears exactly once; a new family must be added to that set.

## Shared GEMM Contract

GEMM uses `A=[M,K]` and `B=[K,N]`. `alpha` applies to the projection, and the final result is divided by a finite positive `output_scale`. `projection(A,B)` interprets inputs according to the following quantization contract:

- `None`: A/B have the same dtype, which is neither E4M3 nor INT8.
- `Fp8UnitScale`: A/B are both E4M3 tensors with the expected scaling already applied.
- `Fp8`: A/B are both E4M3; FP32 `row_scales=[M]` and `channel_scales=[N]` dequantize rows of A and columns of B, respectively.
- `W8A8`: A/B are both INT8 and use FP32 row/channel scales with the same shapes; output is BF16, `K <= 131071`, and the mode applies only to `gemm` and `gemm_bias`.

Quantization occurs before the API call; the current L3 contract does not include dynamic quantization. At least one registered candidate must still support the concrete spec.

<!-- l3-operator:gemm -->
### `gemm`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm(ctx, GemmArgs)` |
| Inputs | `A=[M,K]`, `B=[K,N]`; supports `None`, `Fp8UnitScale`, `Fp8`, and `W8A8` |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = alpha * projection(A,B) / output_scale` |
| Constraints | W8A8 output must be BF16; `WeightVersion` may declare immutable weights and allow prepare to cache an internal prepacked copy |
| Reference test | `gemm_all_candidates_match_torch` |

<!-- l3-operator:gemm_bias -->
### `gemm_bias`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_bias(ctx, GemmBiasArgs { gemm, bias })` |
| Inputs | `A=[M,K]`, `B=[K,N]`, `bias=[N]`; supports all four GEMM quantization contracts |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = (alpha * projection(A,B) + bias) / output_scale`, with bias broadcast across M |
| Constraints | bias dtype matches the projection dtype; W8A8 output must be BF16 |
| Reference test | `gemm_bias_all_candidates_match_torch` |

<!-- l3-operator:gemm_bias_gelu -->
### `gemm_bias_gelu`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_bias_gelu(ctx, GemmBiasGeluArgs { gemm, bias })` |
| Inputs | `A=[M,K]`, `B=[K,N]`, `bias=[N]`; supports `None`, `Fp8UnitScale`, and `Fp8` |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = GELU_tanh(alpha * projection(A,B) + bias) / output_scale` |
| Constraints | bias dtype matches the projection dtype; GELU uses the Torch reference's tanh approximation; W8A8 is unsupported |
| Reference test | `gemm_bias_gelu_all_candidates_match_torch` |

<!-- l3-operator:gemm_geglu -->
### `gemm_geglu`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_geglu(ctx, GemmGegluArgs { gemm })` |
| Inputs | `A=[M,K]`, `B=[K,2N]`; the first N columns of B are `B_gate`, and the last N columns are `B_up`; supports `None` and `Fp8UnitScale` |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = GELU_tanh(alpha*(A@B_gate)) * (alpha*(A@B_up)) / output_scale` |
| Constraints | The second dimension of B is even; FP8 with row/channel scales and W8A8 are unsupported; candidate-specific packing may occur only internally |
| Reference test | `gemm_geglu_all_candidates_match_torch` |

## Shared Attention Contract

Attention computes `softmax(mask(scale * (Q @ K^T))) @ V`. Q/K/V have the same dtype, and `scale` is finite and positive with a default of `1/sqrt(head_dim)`. Dense and KV-cache support MHA, GQA, and MQA and require `query_heads % kv_heads == 0`.

<!-- l3-operator:attention -->
### `attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::attention(ctx, AttentionArgs)` |
| Inputs | `Q=[B,Tq,Hq,D]`, `K/V=[B,Tk,Hkv,D]`; Q/K/V share F16 or BF16 dtype; mask is `None` or `Causal` |
| Output | `Y=[B,Tq,Hq,D]`; ordinary output uses the input dtype, and F16 input may also be written as E4M3 |
| Mathematical semantic | dense scaled dot-product attention |
| Constraints | Causal requires `Tk>=Tq`, with queries aligned to the final positions of the key sequence; ordinary output requires `output_scale=1`; E4M3 stores `round_to_e4m3(attention/output_scale)` |
| Reference test | `attention_all_candidates_match_reference` |

<!-- l3-operator:kv_cache_attention -->
### `kv_cache_attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::kv_cache_attention(ctx, KvCacheAttentionArgs)` |
| Inputs | `Q=[B,Tq,Hq,D]`, `K_cache/V_cache=[B,key_capacity,Hkv,D]`; all share F16 or BF16 dtype; mask is `None` or `Causal` |
| Output | `Y=[B,Tq,Hq,D]`, with the same dtype as the inputs |
| Mathematical semantic | scaled dot-product attention from the query to the first `valid_key_tokens` rows of the cache |
| Constraints | `0<valid_key_tokens<=key_capacity`; when causal, token i is at `query_start+i`, and `query_start+Tq<=valid_key_tokens` is required |
| Reference test | `kv_cache_attention_all_candidates_match_reference` |

<!-- l3-operator:segmented_attention -->
### `segmented_attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::segmented_attention(ctx, SegmentedAttentionArgs)` |
| Inputs | Q/K/V are all `[total_tokens,H,D]` with the same dtype (F16 or BF16); device U32 offsets match the contents of `host_offsets` |
| Output | `Y=[total_tokens,H,D]`, with the same dtype as the inputs |
| Mathematical semantic | independent non-causal self-attention for each segment of a packed token sequence |
| Constraints | offsets contain at least two elements, are monotonically nondecreasing, begin at 0, and end at `total_tokens`; empty segments are allowed; causal and differing Q/KV head counts are unsupported |
| Reference test | `segmented_attention_all_candidates_match_reference` |

## Testing Responsibilities

The catalog test ensures only that semantics are neither missing nor duplicated; it cannot validate the written contracts. A new L3 semantic must also add a semantic test to `src/ops/tests/l3_behavior.rs` and an independent-reference all-candidate numerical test to `src/ops/tests/precision/precision.rs`.

Tests must run through `crates/apxinf-cuda-new/test-new.sh`; a normal `cargo test` from the repository root does not automatically test this crate.

## Prepared tensor operations (experimental `tensor-ops` feature)

`tensor_ops::Context` exposes fixed contiguous tensors and prepared operations for
ResNet/Transformer/U-Net composition. It has separate storage and lifetime tests;
these operations are not yet integrated with the GEMM/Attention persistent recipe
registry or its semantic metadata tests. Do not infer that framework integration
from a successful product build.

`batch_norm(x, parameters, eps)` requires NCHW input and FP32 `[4,C]` parameters
ordered weight, bias, mean, variance. It computes eval BatchNorm with input-centered
scaling; BF16 context rounds its output. `frozen_batch_norm` uses separate FP32
reciprocal-square-root, scale, bias, multiplication and addition steps, and retains
FP32 output even in a BF16 context. Both require finite positive epsilon, matching
context and shape, retain their buffers, and support prepared graph replay.

For prepared tensor convolutions, the experimental process setting
`APXINF_TENSOR_FP32_CONV1D=im2col` selects a preallocated im2col + strict FP32
SGEMM implementation for forward Conv1d represented as NCHW with H=Kh=1,
height stride=1 and height padding=0. Scratch is bounded to 64 MiB per prepared
operation and reused across requests/graph replays. Other shapes, transposed
convolution, BF16/TF32, and oversized scratch use the existing cuDNN path.
Selection is fixed at preparation; it is not a new semantic or persistent recipe.
The unset default remains cuDNN; performance and numerical evidence are separate.


Prepared BF16 convolution candidates (experimental):

- `APXINF_TENSOR_BF16_LAYOUT=reference` selects NCHW storage for RGB stems
  (`Cin=3`) and 1x1 filters. Other BF16 convolutions retain prepared NHWC
  weights. This is a prepare-time layout/algorithm choice, not a model semantic.
- `APXINF_TENSOR_BF16_CONV1D=im2row` lowers forward H=Kh=1 Conv1d to
  BF16 GEMM, padding output position rows to 16. Input conversion is fused
  with lowering; weights are prepared once. Activation/output scratch is
  bounded to 64 MiB, with cuDNN for other profiles or NCHW-selected operations.
  BF16 convolution output is rounded before adding the BF16-rounded bias.
- `Context::with_fp8_conv1d` is an explicit mixed-precision context, currently
  restricted to sm110. It quantizes forward H=Kh=1 Conv1d with Cin/Cout >=128
  and divisible by 16; all other ops retain BF16-context semantics. BF16-rounded
  input and weights use per-tensor E4M3 scales `max(abs(x))/448` (floor 1e-12).
  Weight scale/packing is fixed at prepare, activation scale is updated on GPU
  each run (per batch item); FP32 accumulation writes BF16 before the bias add.
  The native cuBLASLt algorithm, descriptors, scale pointers and <=4 MiB GEMM
  workspace are prepared before capture. Oversized lowering scratch or no
  native candidate returns an error. No external engine or host tensor math.

These paths have separate operator/Graph tests; they remain outside the unified
GEMM/Attention recipe registry. Broad FP8 public-model quality remains
**failed**, not implied by operator correctness or speed. ApexForge v23
separately qualifies one explicit Diffusion layer/timestep plan under its
development numerical and label-quality budget; it is not a general FP8 or
formal release guarantee. The model family owns that plan.

`conv2d_with_precision(..., ConvPrecision::Bf16)` explicitly keeps one
prepared convolution in BF16 inside a BF16/mixed FP8 context. A plain FP32
context rejects this override. `ContextDefault` preserves `conv2d` behavior.
The constraint is lowered as Conv spec p[14], fixed at prepare, retained by
the operation and graph; it does not mutate context precision at replay.
Model layer names and timestep schedules never enter the backend.

`tensor_ops::Activation::Gelu` extends the prepared elementwise path with the
exact-erf formula `0.5*x*(1+erf(x/sqrt(2)))` on FP32 storage. This is distinct
from tanh-approximated GELU. It retains the existing prepare/run/capture lifetime;
there is no allocation, host tensor computation or precision change during replay.
The focused `exact_gelu_matches_erf_golden_and_rebinds_graph_input` test covers
negative tails, zero, positive inputs, input rebinding and retained output.

`Context::conv1d_im2col` selects the prepared FP32 im2col/GEMM provider for
NCHW `[B,C,1,L]` and OIHW `[Cout,Cin,1,K]`, with explicit stride/padding.
Conv spec p[15]=1 records this selection; zero retains existing dispatch.
TF32/BF16 contexts, incompatible shapes and scratch over 64 MiB are rejected.
Scratch and GEMM arguments are prepared once; replay has no provider lookup or
allocation. Kart DP selects this path only for forward temporal convolutions,
leaving transposed and vision convolutions on cuDNN. Other families are unchanged.
The real-shape f64/Graph test covers both library and explicit GEMM providers;
a separate test rejects unsupported precisions. This extends prepared tensor ops,
not the unified recipe registry.
