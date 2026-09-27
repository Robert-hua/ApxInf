pub mod buffer;
#[cfg(feature = "gemm-attention")]
pub mod context;
mod ffi;
#[cfg(feature = "gemm-attention")]
mod graph;
pub mod stream;
#[cfg(feature = "gemm-attention")]
mod workspace;

pub use buffer::{CudaBuffer, CudaDeviceAddress, HostMappedBuffer};
#[cfg(feature = "gemm-attention")]
pub use context::CudaContext;
#[cfg(feature = "gemm-attention")]
pub use graph::{capture, CapturedGraph};
#[cfg(feature = "gemm-attention")]
pub use ops::{
    attention, kv_cache_attention, segmented_attention, AttentionArgs, AttentionMask,
    AttentionPolicy, ExecutionSession, GraphWorkspace, KvCacheAttentionArgs,
    SegmentedAttentionArgs,
};
pub use stream::CudaStream;

#[cfg(feature = "gemm-attention")]
pub mod ops;
#[cfg(feature = "tensor-ops")]
#[path = "ops/tensor_ops/mod.rs"]
pub mod tensor_ops;
