#=  2D scattering transform tests, parameterised by device =#

"""
    run2DTests(; name, toDevice, arrayType, sync)
 
Run the 2D suite on one device. When `arrayType` is not `Array`, every result
is additionally checked against the CPU transform, so the GPU path is verified
for agreement rather than merely for not throwing.
"""
function run2DTests(; name="CPU", toDevice=identity, arrayType=Array,
    sync=() -> nothing)
 
    onGPU = arrayType !== Array
    inputSize = (32, 32, 1, 2)
    Nd, m = 2, 2
 
    st = toDevice(stFlux(inputSize, m))
 
    @testset "$name - construction and accessors" begin
        @test st isa stFlux
        @test ndims(st) == Nd
        @test depth(st) == m
        # three stages per layer, minus the two the final averaging layer omits
        @test length(st.mainChain.layers) == 3 * (m + 1) - 2
        @test length(st.outputSizes) == m + 1
        @test sprint(show, st) isa String
    end
 
    @testset "$name - layers are monogenic" begin
        layers = st.mainChain.layers
        for i in 1:(m+1)
            @test layers[3*i-2] isa MonoConvFFT
            @test ndims(layers[3*i-2]) == Nd
            @test layers[3*i-2].weight isa arrayType
        end
        @test layers[3*m+1].averagingLayer == true
        @test layers[1].averagingLayer == false
    end
 
    @testset "$name - forward pass" begin
        x = randn(Float32, inputSize)
        res = st(toDevice(x))
        sync()
 
        @test res isa ScatteredOut
        @test length(res.output) == m + 1
 
        #=  The sizes stFlux recorded at construction must match what the chain
            actually produces. The most informative assertion here: if 2D is
            subtly mis-wired, recorded and real sizes diverge at this line. =#
        for i in 1:(m+1)
            @test size(res.output[i]) == Tuple(st.outputSizes[i])
        end
 
        @test all(o -> eltype(o) == Float32, res.output)
        @test all(o -> all(isfinite, Array(o)), res.output)
        # zeroth layer is the averaged input: no path dimension
        @test ndims(res.output[1]) == Nd + 2
 
        if onGPU
            @test all(o -> o isa arrayType, res.output)
 
            cpuRes = cpu(st)(x)
            for i in 1:(m+1)
                @test size(cpuRes.output[i]) == size(res.output[i])
                @test Array(res.output[i]) ≈ cpuRes.output[i] atol = 1.0f-3
            end
        end
    end
 
    @testset "$name - flatten" begin
        stf = toDevice(stFlux(inputSize, m, flatten=true))
        x = toDevice(randn(Float32, inputSize))
        out = stf(x)
        sync()
        @test out isa AbstractMatrix
        @test size(out, 2) == inputSize[end]
        onGPU && @test out isa arrayType
        # flattening must not lose or invent coefficients
        res = st(x)
        @test size(out, 1) == sum(prod(size(o)[1:end-1]) for o in res.output)
    end
 
    @testset "$name - normalize = false" begin
        stn = toDevice(stFlux(inputSize, m, normalize=false))
        @test stn(toDevice(randn(Float32, inputSize))) isa ScatteredOut
    end
 
    @testset "$name - batch size mismatch" begin
        x = randn(Float32, 32, 32, 1, 5)   # plan was built for 2 examples
        res = st(toDevice(x))
        sync()
        @test res isa ScatteredOut
        @test all(o -> size(o)[end] == 5, res.output)
        onGPU && @test all(o -> o isa arrayType, res.output)
 
        # an example must transform the same way regardless of how it is batched
        single = st(toDevice(x[:, :, :, 1:2]))
        @test Array(res.output[1])[:, :, :, 1:2] ≈ Array(single.output[1]) atol = 1.0f-3
    end
 
    @testset "$name - gradients" begin
        #=  flatten = true so the loss sums over a plain array rather than a
            ScatteredOut, keeping this about the transform rather than about
            Zygote's handling of the output struct. =#
        stf = toDevice(stFlux(inputSize, m, flatten=true))
        x = toDevice(randn(Float32, inputSize))
        ∇ = Zygote.gradient(t -> sum(abs2, stf(t)), x)[1]
        sync()
        @test size(∇) == size(x)
        @test eltype(∇) == Float32
        @test all(isfinite, Array(∇))
        @test !all(iszero, Array(∇))
        onGPU && @test ∇ isa arrayType
    end
 
    @testset "$name - depth 1" begin
        # the averaging layer immediately follows the first filtering layer
        st1 = toDevice(stFlux(inputSize, 1))
        @test depth(st1) == 1
        res = st1(toDevice(randn(Float32, inputSize)))
        @test length(res.output) == 2
    end
 
    @testset "$name - keyword vocabulary" begin
        #=  The 1D and 2D banks take different keywords, and stFlux forwards
            whatever it gets. These used to surface as MethodErrors from inside
            MonogenicFilterFlux with no indication of the real problem. =#
        @test_throws "no wavelet to configure" stFlux(inputSize, 1, cw=Morlet(π))
        @test_throws "no wavelet to configure" stFlux(inputSize, 1, β=4)
 
        # a ContinuousWavelets boundary means the caller wants the 1D wavelet
        # stage, which does not exist here
        @test_throws "convBoundary" stFlux(inputSize, 1, boundary=PerBoundary())
 
        # `convBoundary` is the one spelling for padding in both dimensions
        stc = toDevice(stFlux(inputSize, 1, convBoundary=FourierFilterFlux.Periodic()))
        @test stc.mainChain.layers[1].bc isa FourierFilterFlux.Periodic
        @test stc(toDevice(randn(Float32, inputSize))) isa ScatteredOut
 
        # monogenic-specific settings still reach the layer
        sts = stFlux(inputSize, 1, scale=3)
        @test sts.mainChain.layers[1].scale == 3
    end
 
    @testset "$name - non-square input" begin
        # a square input hides anywhere the two spatial dimensions get crossed,
        # since transposing is then a no-op
        sz = (32, 24, 1, 2)
        str = toDevice(stFlux(sz, 1))
        res = str(toDevice(randn(Float32, sz)))
        @test size(res.output[1]) == Tuple(str.outputSizes[1])
    end

    @testset "$name -- flatten/roll round trip" begin
        sst = toDevice(stFlux(inputSize, m, poolBy = 3 // 2))
        x = toDevice(randn(Float32, inputSize))
        res = sst(x)
        smooshed = ScatteringTransform.flatten(res)
        sync()
 
        onGPU && @test smooshed isa arrayType
        @test size(smooshed, 2) == inputSize[end]
 
        reconst = roll(smooshed, sst)
        sync()
        for (a, b) in zip(reconst.output, res.output)
            @test size(a) == size(b)
            @test Array(a) ≈ Array(b) atol = 1.0f-3
        end
 
        if onGPU
            # and the flattened vector itself must match the CPU transform
            cpuRes = cpu(sst)(Array(x))
            @test Array(smooshed) ≈ ScatteringTransform.flatten(cpuRes) atol = 1.0f-3
        end
    end
 
    return nothing
end