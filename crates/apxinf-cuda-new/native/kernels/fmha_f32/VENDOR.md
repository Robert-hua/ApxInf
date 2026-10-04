# CUTLASS example 41 forward attention

NVIDIA CUTLASS v3.4.1, commit `bbe579a9e3beb6ea6626d9227ec32d0dae119a49`.
Source: https://github.com/NVIDIA/cutlass/tree/bbe579a9e3beb6ea6626d9227ec32d0dae119a49/examples/41_fused_multi_head_attention
BSD-3-Clause; copyright and license are retained in each header.
The example originated in Meta xFormers. Headers are unmodified; only the
forward kernel is instantiated. Build uses the existing pinned CUTLASS include
tree. The local adapter selects F32 storage, SM80 FastF32 (three TF32 products),
64x64 tiles, head dimension 64, no mask, bias, dropout or backward.
This is an independent prepared tensor provider, not a replacement for FA2 or
other model families. Profiled and numerically checked on Thor sm110.
