module CUDAExt

#=  This extension is intentionally minimal. `stFlux`'s device transfer is
    handled generically by `Functors.@functor stFlux (mainChain,)` plus 
    Flux's own `gpu`/`cpu` - no CUDA-specific code is needed here for that. 
    `ConvFFT`'s actual CUDA-specific behavior already lives entirely in 
    FourierFilterFlux's own CUDAExt, and gets inherited automatically once 
    FourierFilterFlux is loaded.

    This file's real job is triggering activation: `[extensions] CUDAExt =
    ["CUDA", "cuDNN", "cuFFT"]` in Project.toml means all three must be
    loaded before this extension (and therefore this package's CUDA support)
    activates at all, so this `using` line is what actually makes that
    happen, not a formality. It also guarantees FourierFilterFlux's and
    ContinuousWavelets' own CUDA extensions are active by the time anything
    in ScatteringTransform runs on the GPU, since their trigger sets
    (CUDA+cuFFT, and CUDA+cuDNN+cuFFT, respectively) are subsets of this one. =#
using ScatteringTransform, CUDA, cuDNN, cuFFT

end