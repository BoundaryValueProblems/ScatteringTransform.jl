Flux.@functor ScatteredOut
Zygote.@adjoint function ScatteredOut(output) 
    function ∇Scattered(Δ)
        return (Δ.output,)
    end
    ScatteredOut(output), ∇Scattered
end





Zygote.@adjoint function getindex(F::T, i::Integer) where T <: Scattered
    function getInd_rrule(Ȳ)
        zeroNonRefed = map(ii-> ii-1==i ? Ȳ[1] : zeros(eltype(F.output[ii]),
                                                  size(F.output[ii])...),
                           (1:length(F.output)...,)) 
        ∂F = T(zeroNonRefed)
        return ∂F, nothing
    end
    return getindex(F, i), getInd_rrule
end

Zygote.@adjoint function getindex(F::T, inds::AbstractArray) where T<: Scattered
    function getInd_rrule(Ȳ)
        zeroNonRefed = map(ii-> ii-1 in inds ? Ȳ[indexin(ii-1,inds)[1]] :
                           zeros(eltype(F.output[ii]), size(F.output[ii])...),
                           (1:length(F.output)...,)) 
        ∂F = T(zeroNonRefed)
        return ∂F, nothing
    end
    return getindex(F, inds), getInd_rrule
end

Zygote.@adjoint function getindex(x::T, p::pathLocs) where T <: Scattered
    function getInd_rrule(Δ)
        zeroNonRefed = map(ii-> zeros(eltype(x.output[ii]),
                                      size(x.output[ii])...),
                           (1:length(x.output)...,))
        ∂x = T(zeroNonRefed)
        ∂x[p] = Δ
        return ∂x, nothing
    end
    return getindex(x,p), getInd_rrule
end