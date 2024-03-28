#include <metal_stdlib>
#include <metal_math>
using namespace metal;

/**
 * deltaA = exp(einsum('ld, dn -> ldn', delta, A))
 * deltaB_u = einsum('ld, ln, ld -> ldn', delta, B, u)
 */
kernel void discretize(
    device uint& d,
    device uint& n,
    device const float* u,
    device const float* delta,
    device const float* A,
    device const float* B,
    device float* deltaA,
    device float* deltaB_u,
    uint2 gid [[thread_position_in_grid]]
) {
    uint ld_idx = d * gid.y + gid.x;
    float delta_val = delta[ld_idx];
    float delta_u_val = delta_val * u[ld_idx];
    for (uint i = 0; i < n; ++i) {
        uint ldn_idx = d * n * gid.y + n * gid.x + i;
        deltaA[ldn_idx] = exp(delta_val * A[n * gid.x + i]);
        deltaB_u[ldn_idx] = delta_u_val * B[n * gid.y + i];
    }
}
