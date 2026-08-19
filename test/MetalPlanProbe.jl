# test/MetalPlanProbe.jl
#
# Include this from MetalTests.jl BEFORE the model round-trip testset:
#
#     include("MetalPlanProbe.jl")
#
# Purpose: the stFlux round-trip test compares whole arrays, so when it fails it
# tells you "these differ by 18%" and nothing about why. This isolates the plan
# conversion itself. It runs in about a second and, whether it passes or fails,
# prints everything needed to finish the fix without another guess.

using Test, Metal, Adapt, AbstractFFTs, FFTW, FourierFilterFlux

@testset "Metal FFT plan round-trip" begin
    n, ch, batch = 16, 2, 3
    xr = randn(Float32, n, ch, batch)
    xc = complex(xr)

    for (P, x, kind) in ((plan_rfft(xr, 1:1), xr, "r2c"),
                         (plan_fft(xc, 1:1), xc, "c2c"))

        G = Metal.mtl(P)

        # --- what the Metal plan actually looks like -------------------------
        # If the adapt patch guessed wrong about parameter order or property
        # names, this block is the answer.
        @info "plan $kind: cpu -> gpu" typeof(P) typeof(G)
        @info "  sizes" cpuSize = size(P) gpuSize = size(G)
        @info "  gpu properties" propertynames(G)
        if hasproperty(G, :region)
            @info "  regions" cpuRegion = P.region gpuRegion = G.region
        else
            @warn "  gpu plan has no :region property -- adapt will error; " *
                  "use the correct name from propertynames above"
        end

        # Does size() on a Metal r2c plan report the real input length (16) or
        # the complex output length (9)? The patch rebuilds with
        # `zeros(Float32, size(p))`, which is only correct for the former.
        if kind == "r2c"
            @info "  r2c size check" expectedIfInputSize = n expectedIfOutputSize = n ÷ 2 + 1 actual = size(G, 1)
        end

        # --- the invariants that matter --------------------------------------
        Q = adapt(Array, G)
        @info "plan $kind: gpu -> cpu" typeof(Q) size(Q)

        # kind is preserved: an rFFTWPlan must not come back as a cFFTWPlan
        @test typeof(Q) == typeof(P)
        # only the intended axis is transformed
        @test Q.region == P.region
        @test size(Q) == size(P)
        # and the reconstructed plan actually computes the same thing
        @test Q * x ≈ P * x
    end
end
