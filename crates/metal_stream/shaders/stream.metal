// STREAM-style memory bandwidth kernels for Apple Silicon (issue #8 / Phase 3).
// float4 vectorization; host times wall-clock around commit/wait.
#include <metal_stdlib>
using namespace metal;

kernel void stream_copy(
    device const float4* a [[buffer(0)]],
    device float4* c [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id];
}

kernel void stream_scale(
    device const float4* c [[buffer(0)]],
    device float4* b [[buffer(1)]],
    constant float& s [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    b[id] = s * c[id];
}

kernel void stream_add(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}

kernel void stream_triad(
    device const float4* b [[buffer(0)]],
    device const float4* c [[buffer(1)]],
    device float4* a [[buffer(2)]],
    constant float& s [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    a[id] = b[id] + s * c[id];
}
