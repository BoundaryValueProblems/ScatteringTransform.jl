"""
    stFlux(inputSize::NTuple{N}, m=2; outputPool = 2, poolBy = 3//2, 
    σ = abs, normalize = true, flatten = false, trainable = false, kwargs...) 
    where {N}

Create a complete setting of scattering transform that can be applied to a set of
input signals
### Input Arguments
* `inputSize::NTuple{N}`: The size of the input signals. It should be of `NTuple`
type. For example, for a set of `M` 1D signals each of which has length `N`, 
it should be `(N, 1, M)`. For a set of `M` 2D signals each of which is of size 
`N₁` × `N₂`, it should be `(N₁, N₂, 1, M)`. Note that the input signal array 
that will be transformed by the output function from this `stFlux` must have the
same dimension structure as this `inputSize::NTuple{N}`. E.g., if the input 1D
signals are arrange as a matrix of size `(N, M)`, it must be converted to an 
array of size `(N, 1, M)` using `reshape` function before applying the transform.
* `m=2`: The maximum depth of the scattering transform. The default value is `2`.
Note that the depth (or layer) index starts with `0`. So, this scattering transform
sets up `m+1` layers.
* `outputPool = 2`: Subsampling rate after averaging in each layer (it's present
here because averaging removes the high frequencies, so the full signal is inherently
redundant). Can specify different subsampling rates for each direction for 2D
input signals with e.g., `outputPool=(2,3)`. Has a similar structure to poolBy.
* `poolBy = 3//2`: Subsampling rate at each layer for the intermediate representation.
The default value is `3//2` for each layer. Layer-dependent rates can be specified by
supplying `NTuple`, e.g., `(3//2, 2, 3)` for 0th, 1st, and 2nd layers.
* `σ = abs`: The nonlinearity function used. The default function is `abs`, i.e.,
computing the modulus.
* `normalize = true`: If it's `true`, it gives each layer the same average weight
per path, e.g., norm 1 for the 0th layer (since it has only one path), norm 16
for the 1st layer paths (if the first layer has 16 paths), etc. This is primarily
for cases where the classification algorithm needs roughly the same order of 
magnitude variance. 
* `flatten = false`: If false, it returns a result that is a matrix of size 
`(:,nExamples)` where `nExamples` is the last dimension of `inputSize`. If it's
`true`, then return a matrix that is `(nCoefficients,nExamples)`.
* `trainable = false`: If it's `true`, the wavelet filters can be viewed as 
weights (or parameters) to be learned and updated using standard methods from Flux.jl.
For the standard scattering transform scenarios, it should stay as `false`.
* `kwargs...`: come from either the FourierFilterFlux 
[`waveletLayerConstructor`](https://dsweber2.github.io/FourierFilterFlux.jl/dev/constructors/#FourierFilterFlux.waveletLayer), 
or from ContinuousWavlets [`wavelet`](https://dsweber2.github.io/ContinuousWavelets.jl/dev/CWTConstruction/) constructor ContinuousWavelets.
Note that by default, the Morlet wavelet filters are used, but one can change it
by supplying, e.g., `cw=dog2` (Difference of Gaussian or the Mexican Hat Wavelet).

### Output Argument:
* stFlux data structure (also works as a function to apply the scattering transform on
an input signal set) whose main parameters are:
* Nd: Number of spatial dimension of the input signal set
* m: The maximus depth index
* filters: 
* σ: The nonlinearity operation function
* batchSize: Number of the input signals
* normalize: a flag to normalize the output coefficients

# Examples

```jldoctest
julia> using ScatteringTransform

julia> x = [ones(128); zeros(128)];

julia> St = stFlux((256,1,1))
┌ Warning: there are wavelets whose peaks are far enough apart that the trough between them is less than half the height of the highest frequency wavelet
│   minimalRegionComparedToLastPeak = 7.0515
└ @ ContinuousWavelets ~/.julia/packages/ContinuousWavelets/KeITS/src/sanityChecks.jl:33
┌ Warning: there are wavelets whose peaks are far enough apart that the trough between them is less than half the height of the highest frequency wavelet
│   minimalRegionComparedToLastPeak = 4.815
└ @ ContinuousWavelets ~/.julia/packages/ContinuousWavelets/KeITS/src/sanityChecks.jl:33
┌ Warning: there are wavelets whose peaks are far enough apart that the trough between them is less than half the height of the highest frequency wavelet
│   minimalRegionComparedToLastPeak = 3.2091
└ @ ContinuousWavelets ~/.julia/packages/ContinuousWavelets/KeITS/src/sanityChecks.jl:33
stFlux{Nd=1, m=2, filters=[12, 11], σ = abs, batchSize = 1, normalize = true}

julia> St(x)
ScatteredOut{Array{Float32},3} 1 dim. OutputSizes:
    (128, 1, 1)
    (86, 12, 1)
    (57, 11, 12, 1)

```
"""
function stFlux(inputSize::NTuple{N}, m=2; outputPool = 2, poolBy = 3//2, 
    σ = abs, normalize = true, flatten = false, trainable = false, kwargs...) where {N}
    #= N is the length of the NTuple, from which the spatial dimension of input
    signals can be determined. E.g., N=3 <=> 1D signals; N=4 <=> 2D images, etc. =#
    Nd = N - 2
    # Now, setting the outputPool for multi layers
    if length(outputPool) == 1 # replicate for each dimension and layer
        outputPool = ntuple(k -> ntuple(i -> outputPool[1], Nd), m + 1)
    elseif length(outputPool) == m + 1 # replicate for each dimension
        outputPool = ntuple(k -> ntuple(i -> outputPool[k], Nd), m + 1)
    elseif length(outputPool) == Nd # if Nd=1 and outputPool is a scalar, this is bypassed.
        outputPool = ntuple(k -> outputPool, m + 1)
    end
    if typeof(poolBy) <: Union{<:Rational,<:Integer}
        poolBy = map(ii -> RationPool((ntuple(i -> poolBy, Nd)...,),
                nExtraDims=nPathDims(ii + 1) + 1), 1:m+1)
    # RationPool is defined in ~/pool.jl while nExtraDims in ~/shared.jl
    end
    poolBy = makeTuple(m, poolBy) # Defined in ~/shared.jl
    argsToEach = processArgs(m + 1, kwargs) # also in ~/shared.jl
    listOfSizes = [(inputSize..., ntuple(i -> 1, max(i - 1, 0))...) for i = 0:m]
    interstitial = Array{Any,1}(undef, 3 * (m + 1) - 2) 
    #= `interstitial` is an array of 3 functions per layer: 
    1) wavelet filtering; 2) nonlinearity; 3) pooling 
    =#
    for i = 1:m # from Layer 1 to Layer m
        # first transform
        interstitial[3*i-2] = dispatchLayer(listOfSizes[i], Val(Nd); σ=identity,
            argsToEach[i]...)
        #################### TO UPDATE ######################
        if Nd == 1
            nFilters = length(interstitial[3*i-2].weight)
        else
            nFilters = size(interstitial[3 * i - 2].weight)[end]
        end
        #################### TO UPDATE ######################

        pooledSize = poolSize(poolBy[i], listOfSizes[i][1:Nd]) # in ~/pool.jl
        #= then throw away the averaging (we'll pick it up in the actual transform)
        also, in the first layer, merge the channels into the first shearing,
        since max pool isn't defined for arbitrary arrays =#
        if i == 1
            listOfSizes[i+1] = (pooledSize..., (nFilters - 1) * listOfSizes[1][Nd+1], listOfSizes[i][Nd+2:end]...)
            interstitial[3*i-1] = x -> begin
                ax = axes(x)
                return σ.(reshape(x[ax[1:Nd]..., ax[Nd+1][1:end-1], ax[(Nd+2):end]...],
                    (ax[1:Nd]..., listOfSizes[i+1][Nd+1], ax[end])))
            end
        else
            listOfSizes[i+1] = (pooledSize..., (nFilters - 1),
                listOfSizes[i][Nd+1:end]...)
            interstitial[3*i-1] = x -> begin
                ax = axes(x)
                return σ.(x[ax[1:Nd]..., ax[Nd+1][1:end-1], ax[Nd+2:end]...])
            end
        end
        # then pool
        interstitial[3*i] = poolBy[i]
    end

    # final averaging layer
    interstitial[3*m+1] = dispatchLayer(listOfSizes[end], Val(Nd);
        averagingLayer=true, σ=identity, argsToEach[end]...)

    chacha = Chain(interstitial...) # Chain is from `Flux.jl` 
    outputSizes = ([(map(poolSize, outputPool[ii], x[1:Nd])..., x[(Nd+1):end]...) 
                    for (ii, x) in enumerate(listOfSizes)]...,)
    # record the settings used pretty kludgy
    settings = Dict(:outputPool => outputPool, :poolBy => poolBy, :σ => σ, 
        :flatten => flatten, (argsToEach...)...)
    return stFlux{Nd,m,typeof(chacha),typeof(outputSizes),typeof(outputPool),typeof(settings)}(chacha, normalize,
        outputSizes, outputPool, settings)
end

function stFlux(mainChain::C, normalize::Bool, outputSizes::D, 
                outputPool::E, settings::F) where {C,D,E,F}
    # Extract Dimension and Depth from the settings or infer from chain
    Nd = settings[:outputPool] |> first |> length  # spatial dims
    m  = length(outputSizes) - 1                   # depth
    stFlux{Nd, m, C, D, E, F}(mainChain, normalize, outputSizes, outputPool, settings)
end

function dispatchLayer(listOfSizes, Nd::Val{1}; varargs...)
#= For 1D input signals, we use the conventional wavelet filters available in 
our `ContinuousWavelets.jl` package. The following function is defined in yet 
another package of ours: `FourierFilterFlux.jl` that is related to the popluar
`Flux.jl`. =#
    waveletLayer(listOfSizes; varargs...)
end

#=  The 1D and 2D filter banks take different keyword vocabularies, and stFlux
    forwards whatever it was given to whichever one `Nd` selects. Two of those
    differences can cause issues:

      - 1D accepts anything `ContinuousWavelets.wavelet` accepts (`cw`, `β`,
        `averagingLength`, `frameBound`, ...) because `waveletLayer` funnels 
        its `varargs...` there. There is no wavelet in the 2D path at all, the
        monogenic bank is built from the Riesz kernel and Gaussian high/low
        passes, so those arguments have no 2D meaning.

      - Both have a padding/convolution boundary, but under different names.
        `waveletLayer` calls it `convBoundary` and reserves `boundary` for the
        wavelet boundary it hands to `wavelet`. `MonogenicLayer` calls its
        convolution boundary `boundary`. So `boundary = PerBoundary()`, correct
        in 1D, lands in the 2D convolution slot and fails in `effectiveSize`
        three frames down with no hint of the real problem.

    Rather than let those surface as raw MethodErrors from inside
    MonogenicFilterFlux, this is layer will partly remedy the boundary naming 
    issues. `convBoundary` is accepted here as the padding boundary in both 
    dimensions, so a single spelling works either way. =#

# Keywords MonogenicLayer accepts, plus `convBoundary` which is translated below.
const MONOGENIC_KWARGS = (:scale, :boundary, :convBoundary, :init, :dType, :σ,
    :trainable, :plan, :averagingLayer, :Monotype)

# Keywords that only mean something to the 1D wavelet path.
const WAVELET_ONLY_KWARGS = (:cw, :β, :Q, :averagingLength, :averagingType,
    :frameBound, :normalization, :extraOctaves, :p, :s, :decreasing, :fsample,
    :J, :bias)

function dispatchLayer(listOfSizes, Nd::Val{2}; varargs...)
#= For 2D input signals (i.e., images), we use the Monogenic Wavelet Scattering
Networks (MWSNs) of Chak and Saito, from our `MonogenicFilterFlux.jl`. =#
    args = Dict{Symbol,Any}(varargs)

    #=  A ContinuousWavelets boundary means the caller is thinking of the 1D
        wavelet stage, which does not exist here. Say that, rather than letting
        it reach `effectiveSize` as a convolution boundary. =#
    if haskey(args, :boundary) && args[:boundary] isa ContinuousWavelets.WaveletBoundary
        error("""
            `boundary = $(args[:boundary])` is a ContinuousWavelets wavelet boundary, and the 2D
            (monogenic) path has no wavelet stage for it to apply to. If you meant the padding
            used by the convolution, pass `convBoundary` instead, with one of
            FourierFilterFlux's boundary types: `Periodic()`, `Sym()` or `Pad(n...)`.""")
    end

    unknown = setdiff(keys(args), MONOGENIC_KWARGS)
    if !isempty(unknown)
        waveletish = intersect(unknown, WAVELET_ONLY_KWARGS)
        msg = "unsupported keyword argument$(length(unknown) == 1 ? "" : "s") for a 2D " *
              "scattering transform: $(join(sort(collect(unknown)), ", "))."
        if !isempty(waveletish)
            msg *= "\n$(join(sort(collect(waveletish)), ", ")) configure the 1D wavelet " *
                   "filter bank; the 2D path builds a monogenic bank instead and has no " *
                   "wavelet to configure."
        end
        msg *= "\nThe 2D path accepts: scale, Monotype, convBoundary, σ, dType, " *
               "trainable, plan, averagingLayer, init."
        error(msg)
    end

    # one spelling for the padding boundary across both dimensions
    if haskey(args, :convBoundary)
        haskey(args, :boundary) && error(
            "pass either `convBoundary` or `boundary` for the convolution padding, not both")
        args[:boundary] = pop!(args, :convBoundary)
    end

    MonogenicLayer(listOfSizes; args...)
end

Base.size(a::Tuple{AbstractFFTs.Plan,AbstractFFTs.Plan}) = size(a[1])


# actually apply the transform
function (St::stFlux{Dimension,Depth})(x::T) where {Dimension,Depth,T<:AbstractArray}
    #= This actually performs the scattering transform on an input signal set.
    The name of the function is the output structed returned in the above
    `stFlux` function in this file. =#
    mc = St.mainChain.layers
    if ndims(x) < length(size(mc[1]))
        # the input is missing dimensions, so reshape
        targetSize = size(mc[1])
        x = reshape(x, (size(x)..., targetSize[length(targetSize)-ndims(x):end]...))
    end
    if size(x)[end] != getBatchSize(mc[1])
        res = breakAndAdapt(St, x)
    else
        res = applyScattering(mc, x, ndims(St), St, 0)
    end
    if get(St.settings, :flatten, false) # it may not be defined, in which case we don't do it
        batchSize = size(res[1])[end]
        return cat(map(o -> reshape(o, (:, batchSize)), res)..., dims=1)
    else
        return ScatteredOut(res, ndims(mc[1]))
    end
end

_container(x::AbstractArray) = Base.typename(typeof(x)).wrapper
_container(::Type{T}) where {T<:AbstractArray} = Base.typename(T).wrapper

_onDevice(x::AbstractArray) = _container(x) !== Array

function maybeAdapt(reference, x)
    want = _container(reference)
    return want === _container(x) ? x : adapt(want, x)
end

_referenceArray(w::AbstractArray{<:Number}) = w
_referenceArray(w) = _referenceArray(first(w))
_referenceArray(layer::Union{ConvFFT,MonoConvFFT}) = _referenceArray(layer.weight)

"""
    extractAddPadding(x, adr, chunkSize, N)
From `x`, extract the examples `adr` in the last dimension, and make sure that it has a size of `chunkSize`, padding if there are too few examples (this is to make sure the batch size matches).
"""
function extractAddPadding(x, adr, chunkSize, N)
    justUsing = x[fill(Colon(), N)..., :, adr]
    if length(adr) < chunkSize
        actualSize = chunkSize - length(adr)
        return cat(justUsing, fill!(similar(justUsing, size(x)[1:end-1]..., 
                    actualSize), 0), dims=ndims(justUsing)), length(adr)
    else
        return justUsing, chunkSize
    end
end

trim(x, actualSize) = x[axes(x)[1:end-1]..., 1:actualSize]
function breakAndAdapt(St::stFlux{N,D}, x) where {N,D}
    mc = St.mainChain.layers
    cpu_chunk = size(mc[1].fftPlan)[end]
    chunkSize = if _onDevice(x)
        min(size(x)[end], cpu_chunk * 32)  # up to 32x larger chunks on GPU
    else
        cpu_chunk
    end
    nSteps = ceil(Int, (size(x)[end]) / chunkSize) # the first entry is taken care of already
    reference = _referenceArray(St.mainChain[1])
    xAxes = axes(x)
    firstAddr = 1+0:min(size(x)[end], chunkSize)
    firstEx, actualSize = extractAddPadding(x, firstAddr, chunkSize, N) # x[xAxes[1:end-1]..., 1:chunkSize]
    # do the first beforehand to get the sizes
    out = applyScattering(mc, maybeAdapt(reference, firstEx), ndims(St), St, 0)
    # create storage
    # outputs = map(o -> maybeAdapt(typeof(o), zeros(eltype(o), size(o)[1:end-1]..., size(x)[end]...)), out)
    outputs = map(o -> maybeAdapt(typeof(o), fill!(similar(o, size(o)[1:end-1]..., size(x)[end]), zero(eltype(o)))), out)
    # util to write out to outputs at location batchInds
    function writeOut!(out, batchInds, actualSize)
        for jj in 1:length(out)
            mA = maybeAdapt(x, out[jj])
            oAx = axes(outputs[jj])
            @views outputs[jj][oAx[1:end-1]..., batchInds] = trim(mA, actualSize)
        end
    end
    writeOut!(out, firstAddr, actualSize) # write the first batch out
    for ii = 2:nSteps
        out = 3
        # clear output to force garbage collection, otherwise the gpu may be full
        addr = 1+(ii-1)*chunkSize:min(size(x)[end], ii * chunkSize)
        tmpX, actualSize = extractAddPadding(x, addr, chunkSize, N)
        out = applyScattering(mc, maybeAdapt(reference, tmpX), ndims(St), St, 0)
        writeOut!(out, addr, actualSize)
    end
    return outputs
end

function applyScattering(c::Tuple, x, Nd, st, M)
    res = first(c)(x)
    if (typeof(first(c)) <: ConvFFT) || (typeof(first(c)) <: MonoConvFFT)
        tmpRes = res[map(x -> Colon(), 1:Nd)..., end,
            map(x -> Colon(), 1:ndims(res)-Nd-1)...]
        # return a subsampled version of the output at this layer
        poolSizes = (st.outputPool[M+1]..., ntuple(i -> 1, ndims(tmpRes) - Nd - 2)...)
        r = RationPool(st.outputPool[M+1], nExtraDims=nPathDims(M + 1) + 1)
        if st.normalize
            tmpRes = normalize(r(real.(tmpRes)), Nd)
            apld = applyScattering(tail(c), res, Nd, st, M + 1)
            return (tmpRes, apld...)
        else
            tmpRes = r(real.(tmpRes))
            return (tmpRes, applyScattering(tail(c), res, Nd, st, M + 1)...)
        end
    else
        # this is either a reshaping layer or a subsampling layer, so no output
        return applyScattering(tail(c), res, Nd, st, M)
    end
end


applyScattering(::Tuple{}, x, Nd, st, M) = tuple() # all of the returns should
# happen along the way, not at the end

"""
    normalize(x, Nd) -> normedX
normalize `x` over the dimensions `Nd` through `ndims(x)-1`. For example, if `Nd=2`, and `x` is 4D, then `norm(normedX[:,:,:,j], 2) ≈ size(normedX,3)`.
"""
function normalize(x, Nd)
    n = ndims(x)
    totalThisLayer = prod(size(x)[(Nd+1):(n-1)])
    sumSqDims = 1:(n-1)
    normSq = sum(abs.(x) .^ 2, dims=sumSqDims)
    scale = totalThisLayer ./ sqrt.(normSq)
    scale = ifelse.(isnan.(scale) .| isinf.(scale), one(eltype(scale)), scale)
    return x .* scale
end

normalize(sct::ScatteredOut, Nd) = ScatteredOut(map(x -> normalize(x, Nd),
        sct.output), sct.k)

normalize(sct::ScatteredFull, Nd) = ScatteredFull(sct.m, sct.k, sct.data,
    map(x -> normalize(x, Nd), sct.output))