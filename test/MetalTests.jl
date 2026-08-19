const metal_available = @isdefined(Metal) && Metal.functional()

@testset "GPU Tests (Metal)" begin
    if !metal_available
        @warn "No functional Metal found — skipping Metal tests"
    else
        @info "Metal functional — running GPU comparison and timing tests"

        @testset "CPU/GPU consistency, 1D" begin
            init = randn(Float32, 64, 1, 2)
            sst = stFlux(size(init), 2, poolBy=3 // 2)

            resCPU = sst(init)

            sstGPU = gpu(sst)
            initGPU = gpu(init)
            resGPU = sstGPU(initGPU)

            @test typeof(resGPU.output[1]) <: MtlArray

            for (cpuLayer, gpuLayer) in zip(resCPU.output, resGPU.output)
                @test cpuLayer ≈ Array(gpuLayer) atol = 1e-3
            end
        end

        @testset "roll/flatten CPU vs GPU" begin
            initCPU = randn(Float32, 64, 1, 2)
            sst = stFlux(size(initCPU), 2, poolBy=3 // 2)
            resCPU = sst(initCPU)
            smooshedCPU = ScatteringTransform.flatten(resCPU)

            sstGPU = gpu(sst)
            initGPU = gpu(initCPU)
            resGPU = sstGPU(initGPU)
            smooshedGPU = ScatteringTransform.flatten(resGPU)

            @test typeof(smooshedGPU) <: MtlArray
            @test Array(smooshedGPU) ≈ smooshedCPU atol = 1e-3

            reconstCPU = roll(smooshedCPU, sst)
            reconstGPU = roll(smooshedGPU, sstGPU)

            @test all(reconstCPU .≈ resCPU)
            for (cpuLayer, gpuLayer) in zip(reconstGPU.output, resGPU.output)
                @test Array(cpuLayer) ≈ Array(gpuLayer) atol = 1e-3
            end
        end

        @testset "normalize CPU vs GPU" begin
            x = randn(Float32, 10, 4, 3, 5, 7)
            xGPU = gpu(x)

            xpCPU = ScatteringTransform.normalize(x, 2)
            xpGPU = ScatteringTransform.normalize(xGPU, 2)

            @test typeof(xpGPU) <: MtlArray
            @test Array(xpGPU) ≈ xpCPU atol = 1e-3

            for w in eachslice(xpCPU, dims=ndims(x))
                @test norm(w, 2) ≈ 3 * 5
            end
            for w in eachslice(Array(xpGPU), dims=ndims(x))
                @test norm(w, 2) ≈ 3 * 5
            end
        end

        @testset "Gradients CPU vs GPU" begin
            init = randn(Float32, 64, 1, 1)
            initGPU = gpu(init)
            sst = stFlux(size(init), 2, poolBy=3 // 2)
            sstGPU = gpu(sst)

            local ∇CPU_Zeroth, ∇GPU_Zeroth, ∇CPU_First, ∇GPU_First, ∇CPU_Second, ∇GPU_Second
            Metal.@allowscalar begin
                ∇CPU_Zeroth = Zygote.gradient(x -> sst(x)[0][19, 1, 1], init)[1]
                ∇GPU_Zeroth = Zygote.gradient(x -> sstGPU(x)[0][19, 1, 1], initGPU)[1]

                ∇CPU_First = Zygote.gradient(x -> sst(x)[1][11, 5, 1], init)[1]
                ∇GPU_First = Zygote.gradient(x -> sstGPU(x)[1][11, 5, 1], initGPU)[1]

                ∇CPU_Second = Zygote.gradient(x -> sst(x)[2][3, 5, 5, 1], init)[1]
                ∇GPU_Second = Zygote.gradient(x -> sstGPU(x)[2][3, 5, 5, 1], initGPU)[1]
            end

            @test typeof(∇GPU_Zeroth) <: MtlArray
            @test Array(∇GPU_Zeroth) ≈ ∇CPU_Zeroth atol = 1e-3

            @test typeof(∇GPU_First) <: MtlArray
            @test Array(∇GPU_First) ≈ ∇CPU_First atol = 1e-3

            @test typeof(∇GPU_Second) <: MtlArray
            @test Array(∇GPU_Second) ≈ ∇CPU_Second atol = 1e-3
        end

        @testset "RationPool CPU vs GPU" begin
            subsampRates = [3 // 2, 2, 5 // 2, 6 // 5]
            windowSizes = [2, 3, 4]
            @testset "i=$i, s=$s, k=$k" for i in (25, 40), s in subsampRates, k in windowSizes
                x = randn(Float32, i, 2, 3)
                xGPU = gpu(x)
                r = RationPool((s,), k)

                SCPU = r(x)
                SGPU = r(xGPU)
                @test typeof(SGPU) <: MtlArray
                @test Array(SGPU) ≈ SCPU atol = 1e-3

                ∇CPU = Flux.gradient(x -> sum(r(x)), x)[1]
                ∇GPU = Flux.gradient(x -> sum(r(x)), xGPU)[1]
                @test typeof(∇GPU) <: MtlArray
                @test Array(∇GPU) ≈ ∇CPU atol = 1e-3
            end
        end

        @testset "Model & Result conversion round-trips" begin
            init = randn(Float32, 64, 1, 2)
            sst = stFlux(size(init), 2, poolBy=3 // 2)
            resCPU = sst(init)

            @testset "stFlux cpu(gpu(...)) round-trip" begin
                sstRoundTrip = cpu(gpu(sst))
                resRoundTrip = sstRoundTrip(init)
                @test typeof(resRoundTrip.output[1]) <: Array
                for (origLayer, rtLayer) in zip(resCPU.output, resRoundTrip.output)
                    @test origLayer ≈ rtLayer atol = 1e-3
                end
            end

            @testset "ScatteredOut whole-object gpu()/cpu()" begin
                sstGPU = gpu(sst)
                initGPU = gpu(init)
                resGPU_viaModel = sstGPU(initGPU)

                resGPU_viaResult = gpu(resCPU)

                @test typeof(resGPU_viaResult.output[1]) <: MtlArray
                for (l1, l2) in zip(resGPU_viaModel.output, resGPU_viaResult.output)
                    @test Array(l1) ≈ Array(l2) atol = 1e-3
                end

                resRoundTrip = cpu(resGPU_viaResult)
                @test typeof(resRoundTrip.output[1]) <: Array
                for (origLayer, rtLayer) in zip(resCPU.output, resRoundTrip.output)
                    @test origLayer ≈ rtLayer atol = 1e-3
                end
            end
        end

        @testset "Metal FFT plan round-trip" begin
            x = randn(Float32, 16, 2, 3)
            for (P, inp) in ((plan_rfft(x, 1:1), x), (plan_fft(complex(x), 1:1), complex(x)))
                Q = adapt(Array, Metal.mtl(P))
                @test typeof(Q) == typeof(P)
                @test Q.region == P.region
                @test size(Q) == size(P)
                @test Q * inp ≈ P * inp
            end
        end

        @testset "CPU/GPU timing" begin
            sizes = [256, 2048, 16384, 131072]
            nSamples = 5 # Number of runs we test for CPU and GPU timing. 

            for sz in sizes
                GC.gc()

                init = randn(Float32, sz, 1, 1)
                sst = stFlux(size(init), 2, poolBy=3 // 2)
                sstGPU = gpu(sst)
                initGPU = gpu(init)

                sst(init)
                Metal.@sync sstGPU(initGPU)
                GC.gc()

                tCPU = minimum(@elapsed(sst(init)) for _ = 1:nSamples)
                tGPU = minimum(@elapsed(Metal.@sync sstGPU(initGPU)) for _ = 1:nSamples)
                @info "size=$sz" tCPU tGPU speedup = tCPU / tGPU

                sstGPU = nothing
                initGPU = nothing
                GC.gc()
            end
        end
    end
end