module ScatteringTransform

using Core: @__doc__
using ChainRules
using Wavelets, ContinuousWavelets
using Zygote, Flux, LinearAlgebra, AbstractFFTs
using FourierFilterFlux
using MonogenicFilterFlux
using Adapt
using Functors
using RecipesBase
using Base: tail
using ChainRulesCore
using Statistics
using Dates

import Adapt: adapt
import ChainRules: rrule
import Zygote: has_chain_rrule, rrule
import Wavelets: eltypes
import Flux: gpu, cpu


include("shared.jl")
export scatteringTransform, stFlux, depth
include("pathLocs.jl")
export pathLocs
include("scattered.jl")
export Scattered, ScatteredFull, ScatteredOut, nonZeroPaths, cat, addNextPath
include("pool.jl")
export RationPool, nPoolDims, outputsize, poolSize
include("transform.jl")
export gpu, cpu

@doc """
     scatteringTransform(inputSize, m=2, backend::UnionAll=stFlux; kwargs...)
 The constructor for the abstract type scatteringTransform, with the concrete type specified by `backend`.
 """
function scatteringTransform(inputSize, m=2, backend::UnionAll=stFlux; kwargs...)
    backend(inputSize, m; kwargs...)
end

include("utilities.jl")
export getWavelets, flatten, roll, importantCoords, batchOff, getParameters, getMeanFreq, computeLoc
export roll, wrap, flatten, reshapeInputs
include("adjoints.jl")

for f in [:plotOriginalSignal1D, :plotZerothLayer1D, :plotFirstLayer1DSingleWavelet,
          :gifFirstLayer1D, :plotFirstLayer1DAll, :plotFirstLayer1D,
          :plotSecondLayer1DSpecificPath, :gifSecondLayer1DSubset,
          :plotSecondLayer1DFixAndVary, :plotSecondLayer1D, :jointPlot1D,
          :plotOriginalSignal2D, :plotZerothLayer2D, :plotFirstLayer2DSingleWavelet,
          :visualizeFirstLayer2D, :plotFirstLayer2D, :plotFirstLayer2DAll,
          :plotSecondLayer2DSingleWavelet, :visualizeSecondLayer2D, :plotSecondLayer2D]
    @eval function $f(args...; kwargs...)
        ext = Base.get_extension(ScatteringTransform, :ScatteringPlotsExt)
        if isnothing(ext)
            error("Load Plots.jl to use plotting functions: `using Plots`")
        end
        getfield(ext, $(QuoteNode(f)))(args...; kwargs...)
    end
    @eval export $f
end

end # end Module