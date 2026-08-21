module CUDAExt

#=  This extension is intentionally minimal. `stFlux`'s device transfer is
    handled generically by `Functors.@functor stFlux (mainChain,)` plus 
    Flux's own `gpu`/`cpu` - no CUDA-specific code is needed here for that. 
    `ConvFFT`'s actual CUDA-specific behavior already lives entirely in 
    FourierFilterFlux's own CUDAExt, and gets inherited automatically once 
    FourierFilterFlux is loaded.

    This file's real job is triggering activation: `[extensions] CUDAExt =
    ["CUDA", "cuFFT"]` in Project.toml means both must be loaded before this 
    extension (and therefore this package's CUDA support) activates at all. 
    It also guarantees FourierFilterFlux's and ContinuousWavelets' own CUDA 
    extensions are active by the time anything in ScatteringTransform runs on 
    the GPU, since their trigger sets are the same as this one. =#
using ScatteringTransform, CUDA, cuFFT

end