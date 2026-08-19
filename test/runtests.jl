# tests for the various forms of stParallel for the ScatteringTransform
using ScatteringTransform
using ContinuousWavelets
using AbstractFFTs, FFTW
using Test, LinearAlgebra, Statistics, Adapt
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
        haveCUDA = try
            @eval using CUDA, cuFFT
            true
        catch e
            @info "CUDA/cuFFT not available in this environment -- skipping CUDATests.jl" exception=e
            false
        end
        if haveCUDA
            include("CUDATests.jl")
        end
    end

    if GROUP in ("All", "Metal")
        haveMetal = try
            @eval using Metal
            true
        catch e
            @info "Metal.jl not installed -- skipping Metal tests" exception = e
            false
        end
        if haveMetal
            include("MetalTests.jl")
        end
    end
end