# tests for the various forms of stParallel for the ScatteringTransform
using ScatteringTransform
using ContinuousWavelets
using AbstractFFTs, FFTW
using Test, LinearAlgebra, Statistics
using Flux, FourierFilterFlux, MonogenicFilterFlux
using Zygote

#=  GROUP/try-catch/functional() three-layer pattern:
        GROUP=All    (default) attempt every backend this environment has
        GROUP=CUDA   only attempt CUDA -- MetalExt/Metal.jl is never touched
        GROUP=Metal  only attempt Metal -- CUDAExt/CUDA.jl is never touched
        GROUP=CPU    skip both GPU backends entirely =#
const GROUP = get(ENV, "GROUP", "All")

@testset "ScatteringTransform.jl" begin
    include("pathTests.jl")
    include("fluxtests.jl")
    include("2DTests.jl")

    if GROUP in ("All", "CUDA")
        try
            using CUDA, cuDNN, cuFFT
            include("CUDATests.jl")
        catch e
            @info "CUDA/cuDNN/cuFFT not available in this environment -- skipping CUDATests.jl" exception=e
        end
    end

    if GROUP in ("All", "Metal")
        try
            using Metal
            include("MetalTests.jl")
        catch e
            @info "Metal not available in this environment -- skipping MetalTests.jl" exception=e
        end
    end
end