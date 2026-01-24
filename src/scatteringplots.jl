"""
    plotFirstLayer1D(j, origLoc, origSig, index)
Function that plots the first layer gradient wavelet at index `j` across space, along with the original signal. 
It also includes heatmaps of the gradient wavelet in both the spatial and frequency domains. 
The variable `j` specifies which wavelet to plot from the first layer, `index` specifies which example in the batch to plot, 
`origLoc` is the `ScatteredOut` object containing the scattering transform results, and `origSig` is the original input signal. 
"""
function plotFirstLayer1D(j, origLoc, origSig, index=1)
    space = plot(origLoc[1][:, j, index], xlim=(0, length(origLoc[1][:, j, index])+1), legend=false, 
        color=:red, title="First Layer - Gradient Wavelet $j - Varying Location")
    org = plot(origSig[:,:,index], legend=false, color=:red, title="Original Signal", xlim=(0, length(origSig[:,:,index])+1))
    ∇h = heatmap(origLoc[1][:, j, index]', xlabel="space", 
        yticks=false, ylabel="", title="First Layer gradient - Wavelet j=$j")
    ∇̂h = heatmap(log.(abs.(rfft(origLoc[1][:, j, index], 1)) .^ 2)', xlabel="frequency", 
        yticks=false, ylabel="", title="Log-power Frequency Domain - Wavelet j=$j")
    l = Plots.@layout [a; b{0.1h}; [b c]]
    plot(space, org, ∇h, ∇̂h, layout=l, size=(1200, 800), margin=5Plots.mm)
end

"""
    gifFirstLayer(origLoc, origSig, saveTo="tmp.gif", fps = 2, index)
Function to create a GIF visualizing all wavelets in the first layer across space for each example in the batch. 
The variable `origLoc` is the `ScatteredOut` object containing the scattering transform results, `index` specifies which example in the batch to plot, 
`origSig` is the original input signal, `saveTo` specifies the file path to save the GIF, and `fps` sets the frames per second for the GIF animation. 
"""
function gifFirstLayer(origLoc, origSig, saveTo="gradientFigures/tmp.gif", fps=2, index=1)
    anim = Animation()
    for j = 1:size(origLoc[1])[end-1]
        plotFirstLayer1D(j, origLoc, origSig, index)
        frame(anim)
    end
    gif(anim, saveTo, fps=fps)
end

"""
    plotFirstLayer1DAll(index, origLoc, origSig, saveTo="gradientFigures/tmp2.png", cline=:darkrainbow)
Function that plots all first layer gradient wavelets for a specific example signal `index` across space, along with the original signal. 
It also includes heatmaps of the gradient wavelets in both the spatial and frequency domains. 
The variable `index` specifies which example in the batch to plot, `origLoc` is the `ScatteredOut` object 
containing the scattering transform results, `origSig` is the original input signal, and `saveTo` is the file path to save the plot.
"""
function plotFirstLayer1DAll(index, origLoc, origSig, saveTo="gradientFigures/tmp2.png", cline=:darkrainbow)
    space = plot(origLoc[1][:, :, index], line_z=(1:size(origLoc[1], 2))', xlim=(0, length(origLoc[1][:, 1, index])+1), 
        legend=false, colorbar=true, color=cline, title="first layer gradient wavelets")
    org = plot(origSig[:,:,index], legend=false, color=:red, title="Original Signal", xlim=(0, length(origSig[:,:,index])+1))
    ∇h = heatmap(origLoc[1][:, 1:end, index]', xlabel="space",
        ylabel="wavelet index", title="First Layer gradients")
    ∇̂h = heatmap(log.(abs.(rfft(origLoc[1][:, 1:end, index], 1)) .^ 2)', xlabel="frequency",
        ylabel="wavelet index", title="Log-power Frequency Domain")
    l = Plots.@layout [a; b{0.1h}; [b c]]
    plt = plot(space, org, ∇h, ∇̂h, layout=l, size=(1200, 800), margin=5Plots.mm)
    savefig(plt, saveTo)
end

"""
    plotFirstLayer(stw, St, saveTo="gradientFigures/firstLayer.png", index=1)
Function that creates a heatmap of the first layer scattering transform results at a specified example index. 
The variable `stw` is the scattered output, `St` is the scattering transform object, `saveTo` is the file 
path to save the plot, and `index` specifies which example in the batch to plot.
"""
function plotFirstLayer(stw, St, saveTo="gradientFigures/firstLayer.png", index=1)
    f1, f2, f3 = getMeanFreq(St) # the mean frequencies for the wavelets in each layer. 
    plt = heatmap(1:size(stw[1], 1), f1[1:end-1], stw[1][:, :, index]', 
            xlabel="time index", ylabel="Frequency (Hz)", 
            color=:viridis, title="First Layer", size=(1200, 800))
    savefig(plt, saveTo)
end


meanWave(wave) = sum(real.(range(0, stop=1, length=size(wave, 1)) .* wave), dims=1) ./ sum(real.(wave), dims=1)

# As far as I can tell, this function is not used elsewhere. 
"""
    plotSecondLayer1D(loc, origLoc, wave1, wave2, original=false, subsamSz=(128,85,))
"""
function plotSecondLayer1D(loc, origLoc, wave1, wave2, original=false, subsamSz=(128, 85,), c=:thermal, lastDiagFreq=true)
    waveUsed = real.(ifftshift(irfft(wave1[:, loc[2]], subsamSz[1] * 2)))
    l1wave = plot(waveUsed, legend=false, titlefontsize=8, title="layer 1 ($(loc[2]))")
    annotate!(size(waveUsed, 1) * 5 / 6, maximum(waveUsed), Plots.text("freq = $(meanWave(wave1)[loc[2]])"[1:13], 5))
    waveUsed = real.(ifftshift(irfft(wave2[:, loc[1]], subsamSz[2] * 2)))
    l2wave = plot(waveUsed, legend=false, titlefontsize=8, title="layer 2 ($(loc[1]))")
    annotate!(size(waveUsed, 1) * 5 / 6, maximum(waveUsed),
        Plots.text("freq = $(meanWave(wave2)[loc[1]])"[1:13], 5))
    if original != false
        org = plot([original original], line_z=(1:size(origLoc[1], 2))', legend=false, colorbar=true, color=:darkrainbow) # todo: including the colorbar here is a real hack to get them to line up
    end
    ∇heat = heatmap(origLoc[2][:, :, loc...]', ylabel="wavelet location",
        xlabel="space", color=c, title="second layer gradient ")
    if lastDiagFreq
        ∇plt = heatmap(log.(abs.(rfft(origLoc[2][:, :, loc...], 1))), xlabel="wavelet location", ylabel="frequency", color=c, title="varying location, path $(loc[2:-1:1])")
    else
        ∇plt = plot(origLoc[2][:, :, loc...], line_z=(28:-1:1)', legend=false, color=:cividis, title="varying location, path $(loc[2:-1:1])")
    end

    if original != false
        l = Plots.@layout [[a; b{0.1h}; [c d]] e]
        return plot(∇heat, org, l1wave, l2wave, ∇plt, layout=l)
    else
        l = Plots.@layout [[a; [b c]] d]
        return plot(∇heat, l1wave, l2wave, ∇plt, layout=l)
    end
end

"""
    plotSecondLayerSpecificPath(stw, St, firstLayerWaveletIndex, secondLayerWaveletIndex, original, index=1)
`stw` is the scattered output, `St` is the scattering transform object, `firstLayerWaveletIndex` and `secondLayerWaveletIndex` specify the path to plot, `original` is the original signal, and `index` specifies 
which example in the batch to plot. This function creates a plot showing the original signal and the scattering result for the specified path. It also displays the mean frequencies associated with the 
selected wavelets. Finally, it displays the log-power norm of the second layer signal for the specified path. This value is used elsewhere to create heatmaps of the second layer scattering results. 
"""
function plotSecondLayerSpecificPath(stw, St, firstLayerWaveletIndex, secondLayerWaveletIndex, original, index=1)
    # Plot of original signal. 
    org = plot(original[:,:,index], legend=false, color=:red, title="Original Signal", xlabel="time (samples)", ylabel="amplitude", xlims=(0, length(original[:,:,index])+1))
    f1, f2, f3 = getMeanFreq(St)
    
    # Plot the signal for a specific path. 
    signalLayer1Freq = f1[firstLayerWaveletIndex]; signalLayer2Freq = f2[secondLayerWaveletIndex]
    titlePlot = plot(title="Path: First Layer - wavelet $firstLayerWaveletIndex, Second Layer - wavelet $secondLayerWaveletIndex\n" *
                            "First Layer Freq = $(round(signalLayer1Freq, sigdigits=3)) Hz | " * 
                            "Second Layer Freq = $(round(signalLayer2Freq, sigdigits=3)) Hz",
                     grid=false, showaxis=false, xticks=nothing, yticks=nothing, bottom_margin=-5Plots.px, titlefontsize=11)
    
    path_spatial = stw[2][:, secondLayerWaveletIndex, firstLayerWaveletIndex, index]
    ∇h = plot(path_spatial, xlabel="time (samples)", ylabel="amplitude", title="Second Layer Plot", legend=false, linewidth=1.5, frame=:box, fill=0, fillalpha=0.5, 
              xlims=(0, length(path_spatial)), ylims=(0, maximum(path_spatial)*1.01))

    secondLayerNorm = log10.(norm(stw[2][:, secondLayerWaveletIndex, firstLayerWaveletIndex, index]))
    normPlot = plot(title="Second Layer Signal norm (log-power) = $(round(secondLayerNorm, sigdigits=4))",
                    grid=false, showaxis=false, xticks=nothing, yticks=nothing, titlefontsize=11, framestyle=:none)
    l = Plots.@layout [a{0.4h}; title{0.05h}; c; d{0.05h}]
    plot(org, titlePlot, ∇h, normPlot, layout=l, margin=4Plots.mm, size=(900,600))
end

"""
    plotSecondLayer1DSubsetGif(stw, St, firstLayerWavelets, secondLayerWavelets, original, saveTo="secondLayerFigures/tmp2.gif", fps=2, index=1)
Create a GIF visualizing the second layer scattering results for specified subsets of wavelets from the first and second layers. The variables `firstLayerWavelets` and `secondLayerWavelets` are arrays 
containing the indices of the wavelets to be visualized from the first and second layers, respectively. For example, to visualize all the wavelets from the first layer with respect to a specific 
wavelet from the second layer, you can set `firstLayerWavelets = 1:size(stw[1], 2)` and `secondLayerWavelets = k`, where `k` is the index of the desired second layer wavelet. Once again, the `index` 
parameter specifies which example in the batch to plot. It defaults to the first example in the batch. 
"""
function plotSecondLayer1DSubsetGif(stw, St, firstLayerWavelets, secondLayerWavelets, original, saveTo="secondLayerFigures/tmp2.gif", fps=2, index=1)
    anim = Animation()
    for j in firstLayerWavelets, k in secondLayerWavelets
        plotSecondLayerSpecificPath(stw, St, j, k, original, index)
        frame(anim)
    end
    gif(anim, saveTo, fps=fps)
end

"""
    plotSecondLayerFixAndVary(stw, St, firstLayerWavelets, secondLayerWavelets, saveTo="secondLayerFigures/sliceByLayer.gif", fps=2, index=1)
Create a GIF visualizing slices of the second layer scattering results by fixing one layer's wavelet and varying the other layer's wavelet. 
The variables `firstLayerWavelets` and `secondLayerWavelets` are arrays containing the indices of the wavelets to be visualized from the first and second layers, respectively. 
If `firstLayerWavelets` contains only one index, the function fixes that wavelet and varies the second layer wavelets, and vice versa.
"""
function plotSecondLayerFixAndVary(stw, St, firstLayerWavelets, secondLayerWavelets, saveTo="secondLayerFigures/sliceByLayer.gif", fps=2, index=1)
    f1, f2, f3 = getMeanFreq(St)
    anim = Animation()

    if length(firstLayerWavelets) == 1
        # Fixed first layer, vary second layer. 
        fixedFirstIdx = firstLayerWavelets[1]
        for jj in secondLayerWavelets
            if jj > size(stw[2], 2)
                continue
            end
            toPlot = stw[2][:, jj, :, index]
            heatmap(1:size(toPlot, 1), f1[1:end-1], toPlot', title="Second Layer Wavelet $jj, Frequency=$(round(f2[jj],sigdigits=4))Hz", 
                    xlabel="time (samples)", ylabel="First Layer Frequency (Hz)", c=cgrad(:viridis, scale=:exp))
            frame(anim)
        end
    else
        # Fixed second layer, vary first layer.
        for jj in firstLayerWavelets
            if jj > size(stw[2], 3)
                continue
            end
            toPlot = stw[2][:, :, jj, index]
            heatmap(1:size(toPlot, 1), f2[1:end-1], toPlot', title="First Layer Wavelet $jj, Frequency=$(round(f1[jj],sigdigits=4))Hz", 
                    xlabel="time (samples)", ylabel="Second Layer Frequency (Hz)", c=cgrad(:viridis, scale=:exp))
            frame(anim)
        end
    end
    gif(anim, saveTo, fps=fps)
end

"""
    plotSecondLayer(stw; title="Second Layer results", xVals=-1, yVals=-1, logPower=true, toHeat=nothing, c=cgrad(:viridis, [0,.9]), threshold=0, linePalette=:greys, minLog=NaN, kwargs...)
TODO fix the similarity of these names.
xVals and yVals give the spacing of the grid, as it doesn't seem to be done
correctly by default. xVals gives the distance from the left and the right
as a tuple, while yVals gives the distance from the top and the bottom,
also as a tuple. Default values are `xVals = (.037, .852), yVals = (.056, .939)`, or if you have no title, use `xVals = (.0105, .882), yVals = (.056, .939)`
If you have no colorbar, set `xVals = (.0015, .997), yVals = (.002, .992)`
In the case that arbitrary space has been introduced, if you have a title, use `xVals = (.037, .852), yVals = (.056, .939)`, or if you have no title, use `xVals = (.0105, .882), yVals = (.056, .939)`
"""
function plotSecondLayer(stw::ScatteredOut, St, index; kwargs...)
    secondLayerRes = stw[2]
    if ndims(secondLayerRes) > 3
        return plotSecondLayer(secondLayerRes[:, :, :, index], St; kwargs...)
    else
        return plotSecondLayer(secondLayerRes, St; kwargs...)
    end
end

function plotSecondLayer(stw, St; title="Second Layer results", xVals=-1, yVals=-1, logPower=true, toHeat=nothing, c=cgrad(:viridis, [0, 0.9]), threshold=0, freqsigdigits=3, linePalette=:greys, 
                         minLog=NaN, subClims=(Inf, -Inf), δt=1000, firstFreqSpacing=nothing, secondFreqSpacing=nothing, transp=true, labelRot=30, xlabel=nothing, ylabel=nothing, frameTypes=:box, miniFillAlpha=0.5, kwargs...)
    n, m = size(stw)[2:3]
    freqs = getMeanFreq(St, δt)
    freqs = map(x -> round.(x, sigdigits=freqsigdigits), freqs)[1:2]
    gr(size=2.5 .* (280, 180))
    if !(typeof(c) <: PlotUtils.ContinuousColorGradient)
        c = cgrad(c)
    end
    if toHeat == nothing
        toHeat = [norm(stw[:, i, j, 1]) for i = 1:n, j = 1:m]
    end
    if firstFreqSpacing == nothing
        firstFreqSpacing = 1:m
    end
    if secondFreqSpacing == nothing
        secondFreqSpacing = 1:n
    end
    # transp means transpose so that the second layer frequency is along the x-axis
    if transp
        xTicksFreq = (secondFreqSpacing, freqs[2][secondFreqSpacing])
        yTicksFreq = (firstFreqSpacing, freqs[1][firstFreqSpacing])
        freqs = reverse(freqs)
        nTmp = n
        n = m
        m = nTmp
        toHeat = toHeat'
        xInd = 2
        yInd = 1
        stw = permutedims(stw, (1, 3, 2, (4:ndims(stw))...))
    else
        xTicksFreq = (firstFreqSpacing, freqs[1][firstFreqSpacing])
        yTicksFreq = (secondFreqSpacing, freqs[2][secondFreqSpacing])
        (1:m, freqs[1])
        xInd = 1
        yInd = 2
    end
    if xVals == -1 && title == ""
        xVals = (0.0105, 0.882)
    elseif xVals == -1
        xVals = (0.002, 0.880)
    end
    Δx = xVals[2] - xVals[1]
    xrange = range(xVals[1] + Δx / m - Δx / (m + 3),
        stop=xVals[2] - 2 * Δx / (m + 3) + Δx / m, length=m)
    if yVals == -1
        yVals = (0.0, 0.995)
    end
    Δy = yVals[2] - yVals[1]
    yrange = range(yVals[1] + Δy / n - Δy / (n + 3), stop=yVals[2] - 2 * Δy / (n + 3) + Δy / n, length=n)
    if logPower
        toHeat = log10.(toHeat)
        if !isnan(minLog)
            toHeat = max.(minLog, toHeat)
        end
    end
    bottom = min(minimum(toHeat), subClims[1])
    top = max(subClims[2], maximum(toHeat))
    totalRange = top - bottom
    # substitute given x and y labels if needed
    if isnothing(xlabel)
        xlabel = "Layer $(xInd) frequency (Hz)"
    end
    if isnothing(ylabel)
        ylabel = "Layer $(yInd) frequency (Hz)"
    end

    if title == ""
        plt = heatmap(toHeat; yticks=yTicksFreq, xticks=xTicksFreq, tick_direction=:out, rotation=labelRot,
            xlabel=xlabel, ylabel=ylabel, c=c, clims=(bottom, top), kwargs...)
    else
        plt = heatmap(toHeat; yticks=yTicksFreq, xticks=xTicksFreq, tick_direction=:out, rotation=labelRot,
            title=title, xlabel=xlabel, ylabel=ylabel, c=c, clims=(bottom, top), kwargs...)
    end
    nPlot = 2
    for i in 1:n, j in 1:m
        if maximum(abs.(stw[:, i, j, :])) > threshold
            plt = plot!(stw[:, i, j, :], legend=false, subplot=nPlot,
                bg_inside=c[(toHeat[i, j]-bottom)/totalRange],
                ticks=nothing, palette=linePalette, frame=frameTypes,
                inset=(1, bbox(xrange[j], yrange[i], Δx / (m + 10),
                    Δy / (n + 10), :bottom, :left)), fill=0, fillalpha=miniFillAlpha)
            nPlot += 1
        end
    end
    plt
end

"""
    jointPlot(thingToPlot, thingName, cSymbol, St; sharedColorScaling=:exp, targetExample=1, δt=1000, freqigdigits=3, sharedColorbar=false, extraPlot=nothing, allPositive=false, logPower=false)
Create a joint plot visualizing the zeroth, first, and second layer scattering results for a specified example. The variable `thingToPlot` is a tuple containing the scattering results for the 
zeroth, first, and second layers, `thingName` is the title for the plot, `cSymbol` specifies the color gradient to use, and `St` is the scattering transform object. The function allows for 
various customization options, including shared color scaling, target example selection, frequency digit rounding, and additional plotting options. 
"""
function jointPlot(thingToPlot, thingName, cSymbol, St; sharedColorScaling=:exp, targetExample=1, δt=1000, freqigdigits=3, sharedColorbar=false, extraPlot=nothing, allPositive=false, logPower=false, kwargs...)
    if sharedColorbar
        clims = (min(minimum.(thingToPlot)...), max(maximum.(thingToPlot)...))
        climszero = clims
        climsfirst = clims
        climssecond = clims
        toHeat = [norm(thingToPlot[2][:, i, j, targetExample], Inf) for i = 1:size(thingToPlot[2], 2), j = 1:size(thingToPlot[2], 3)]
    else
        climszero = (min(minimum.(thingToPlot[0])...), max(maximum.(thingToPlot[0])...))
        climsfirst = (min(minimum.(thingToPlot[1])...), max(maximum.(thingToPlot[1])...))
        climssecond = (min(minimum.(thingToPlot[2])...), max(maximum.(thingToPlot[2])...))
        toHeat = [norm(thingToPlot[2][:, i, j, targetExample], Inf) for i = 1:size(thingToPlot[2], 2), j = 1:size(thingToPlot[2], 3)]
    end
    firstLay = thingToPlot[1][:, :, targetExample]'
    zeroLay = thingToPlot[0][:, :, targetExample]'
    toHeat[toHeat.==0] .= -Inf    # we would like zeroes to not actually render
    firstLay[firstLay.==0] .= -Inf # for either layer

    # adjust the other parts to be log if logPower is true
    if logPower && allPositive
        absThing = map(x -> abs.(x), (thingToPlot[0], thingToPlot[1], thingToPlot[2]))
        clims = (min(minimum.(absThing)...), max(maximum.(absThing)...))
        firstLay = log10.(firstLay)
        zeroLay = log10.(abs.(zeroLay))
        climszero = log10.(clims)
        climsfirst = log10.(clims)
        climssecond = log10.(clims)
    elseif logPower
        error("not currently plotting log power and negative values")
    end

    if allPositive
        c = cgrad(cSymbol, scale=sharedColorScaling)
        cSecond = cgrad(cSymbol)
    else
        zeroAt = -climssecond[1] / (climssecond[2] - climssecond[1]) # set the mid color switch to zero
        c = cgrad(cSymbol, [0, zeroAt], scale=sharedColorScaling)
        cSecond = cgrad(cSymbol, [0, zeroAt])
    end

    # define the spatial locations as they correspond to the input
    spaceLocs = range(1, size(St)[1], length=length(zeroLay))

    p2 = plotSecondLayer(thingToPlot[2][:, :, :, targetExample], St; title="Second Layer", toHeat=toHeat, logPower=logPower, c=c, clims=climssecond, subClims=climssecond, cbar=false, xVals=(0.000, 0.993), yVals=(0.0, 0.994), transp=true, xlabel="", kwargs...)
    freqs = getMeanFreq(St, δt)
    freqs = map(x -> round.(x, sigdigits=freqigdigits), freqs)
    p1 = heatmap(firstLay, c=c, title="First Layer", clims=climsfirst, cbar=false, yticks=((1:size(firstLay, 1)), ""), xticks=((1:size(firstLay, 2)), ""), bottom_margin=-10Plots.px)
    p0 = heatmap(spaceLocs, 1:1, zeroLay, c=c, xlabel="time (ms)\nZeroth Layer", clims=climszero, cbar=false, yticks=nothing, top_margin=-10Plots.px, bottom_margin=10Plots.px)
    colorbarOnly = scatter([0, 0], [0, 1], zcolor=[0, 3], clims=climssecond, xlims=(1, 1.1), xshowaxis=false, yshowaxis=false, label="", c=c, grid=false, framestyle=:none)
    if extraPlot == nothing
        # extraPlot = scatter([0,0], [0,1], legend=false, grid=false, x="Layer 2 frequency", foreground_color_subplot=:white, top_margin=-10Plots.px, showaxis=false, yticks=nothing)
        extraPlot = plot(xlabel="Layer 2 frequency (Hz)", grid=false, xticks=(1:5, ""), showaxis=false, yticks=nothing, bottom_margin=-10Plots.px, top_margin=-20Plots.px)
        # extraPlot = heatmap(zeroLay, c=c, xlabel="location\nZeroth Layer", clims=climszero, cbar=false, yticks=nothing, top_margin=-10Plots.px, bottom_margin=10Plots.px)
    end
    titlePlot = plot(title=thingName, grid=false, showaxis=false, xticks=nothing, yticks=nothing, bottom_margin=-10Plots.px)
    lay = Plots.@layout [o{0.00001h}; [[a b; c{0.1h} d{0.1h}] b{0.04w}]]
    plot(titlePlot, p2, p1, extraPlot, p0, colorbarOnly, layout=lay, size=(1200,800))
end
