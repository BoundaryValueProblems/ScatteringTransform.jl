module MetalExt

using ScatteringTransform, Metal

#= NNlib ships no Metal pooling kernel, so meanpool! falls through to 
meanpool_direct!, which scalar-indexes and is therefore rejected on MtlArray.
cuDNN provides the equivalent on the CUDA side, so this only shows up on Metal.

    NNlib normalises 3-D and 4-D pooling to the 5-D method, so implementing 
the 5-D case intercepts every call. Mean pooling is prod(kernel) strided reads
and an accumulate: for a fixed window offset the source indices are strided and 
non-overlapping, so this is race-free without atomics.

    This is type piracy in the strict sense, as neither NNlib.meanpool! nor 
MtlArray is ours, but it is the same shape as what NNlibCUDA does for CUDA, and 
it lives in a Metal-triggered extension. =#

using NNlib
using NNlib: PoolDims, kernel_size, stride, dilation, padding, output_size

function _mtlZeroPad(x::MtlArray{T,5}, p::NTuple{6,Int}) where {T}
    all(iszero, p) && return x
    s = size(x)
    xp = similar(x, T, (s[1] + p[1] + p[2], s[2] + p[3] + p[4],
                        s[3] + p[5] + p[6], s[4], s[5]))
    fill!(xp, zero(T))
    @views xp[p[1].+(1:s[1]), p[3].+(1:s[2]), p[5].+(1:s[3]), :, :] .= x
    return xp
end
 
# indices of the window-offset `c` (0-based) along one spatial axis
_poolIdx(c, st, dil, nOut) = (1 + c * dil):st:(1 + c * dil + st * (nOut - 1))
 
function NNlib.meanpool!(y::MtlArray{T,5}, x::MtlArray{T,5}, pdims::PoolDims;
                         alpha = true, beta = false, kwargs...) where {T}
    kern = kernel_size(pdims)
    st = stride(pdims)
    dil = dilation(pdims)
    o1, o2, o3 = output_size(pdims)
    xp = _mtlZeroPad(x, padding(pdims))
 
    acc = similar(y)
    fill!(acc, zero(T))
    for c3 = 0:kern[3]-1, c2 = 0:kern[2]-1, c1 = 0:kern[1]-1
        i1 = _poolIdx(c1, st[1], dil[1], o1)
        i2 = _poolIdx(c2, st[2], dil[2], o2)
        i3 = _poolIdx(c3, st[3], dil[3], o3)
        @views acc .+= xp[i1, i2, i3, :, :]
    end
    acc ./= T(prod(kern))
 
    if iszero(beta)
        y .= T(alpha) .* acc
    else
        y .= T(alpha) .* acc .+ T(beta) .* y
    end
    return y
end
 
function NNlib.∇meanpool!(dx::MtlArray{T,5}, dy::MtlArray{T,5}, y::MtlArray{T,5},
                          x::MtlArray{T,5}, pdims::PoolDims;
                          alpha = true, beta = false, kwargs...) where {T}
    kern = kernel_size(pdims)
    st = stride(pdims)
    dil = dilation(pdims)
    p = padding(pdims)
    o1, o2, o3 = output_size(pdims)
 
    scaled = dy ./ T(prod(kern))
    padded = !all(iszero, p)
    s = size(dx)
    dxp = padded ?
          fill!(similar(dx, T, (s[1] + p[1] + p[2], s[2] + p[3] + p[4],
                                s[3] + p[5] + p[6], s[4], s[5])), zero(T)) :
          fill!(dx, zero(T))
 
    # each offset writes a strided, non-overlapping set of destinations, so the
    # accumulate is safe; the overlap between offsets is serialised by the loop. 
    for c3 = 0:kern[3]-1, c2 = 0:kern[2]-1, c1 = 0:kern[1]-1
        i1 = _poolIdx(c1, st[1], dil[1], o1)
        i2 = _poolIdx(c2, st[2], dil[2], o2)
        i3 = _poolIdx(c3, st[3], dil[3], o3)
        @views dxp[i1, i2, i3, :, :] .+= scaled
    end
 
    if padded
        @views dx .= dxp[p[1].+(1:s[1]), p[3].+(1:s[2]), p[5].+(1:s[3]), :, :]
    end
    if !iszero(alpha) && alpha != true
        dx .*= T(alpha)
    end
    return dx
end

end