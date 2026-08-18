"""
    plotOriginalSignal1D(f; title="Original Signal", saveTo=nothing, index=1)
Function that plots the original signal at a specified example index. 
"""
function plotOriginalSignal1D(f; title="Original Signal", saveTo=nothing, index=1)
    plt = plot(f[:,1,index], title=title, legend=false, xlim=(0, length(f[:, 1, index])+1), color=:blue, margin=5Plots.mm, size=(720,480))
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
    plotZerothLayer1D(sf; title="Zeroth Layer", saveTo=nothing, index=1)
Function that plots the zeroth layer of the scattering transform at a specified example index. 
"""
function plotZerothLayer1D(sf; title="Zeroth Layer", saveTo=nothing, index=1)
    plt = plot(sf[0][:, 1, index], title=title, legend=false, xlim=(0, length(sf[0][:, 1, index])+1), color=:blue, margin=5Plots.mm, size=(720,480))
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
    plotFirstLayer1DSingleWavelet(j, origLoc, origSig; title="First Layer", saveTo=nothing, index=1)
The variable `j` specifies which wavelet results to plot from the first layer, `index` specifies which example in the batch to plot, 
`origLoc` is the `ScatteredOut` object containing the scattering transform results, and `origSig` is the original input signal. 
It also includes heatmaps of the gradient wavelet in both the spatial and frequency domains. 
"""
function plotFirstLayer1DSingleWavelet(j, origLoc, origSig; title="First Layer", saveTo=nothing, index=1)
    space = plot(origLoc[1][:, j, index], xlim=(0, length(origLoc[1][:, j, index])+1), legend=false, 
        color=:red, title="$title - Gradient Wavelet $j - Varying Location")
    org = plot(origSig[:,:,index], legend=false, color=:red, title="Original Signal", xlim=(0, length(origSig[:,:,index])+1))
    ∇h = heatmap(origLoc[1][:, j, index]', xlabel="space", 
        yticks=false, ylabel="", title="First Layer gradient - Wavelet j=$j")
    ∇̂h = heatmap(log.(abs.(rfft(origLoc[1][:, j, index], 1)) .^ 2)', xlabel="frequency", 
        yticks=false, ylabel="", title="Log-power Frequency Domain - Wavelet j=$j")
    l = Plots.@layout [a; b{0.1h}; [b c]]
    plt = plot(space, org, ∇h, ∇̂h, layout=l, size=(1280, 720), margin=5Plots.mm)
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
    gifFirstLayer1D(origLoc, origSig; fps=2, title="First Layer", saveTo=nothing, index=1)
Function to create a GIF visualizing all wavelets in the first layer across space for each example in the batch. 
The variable `origLoc` is the `ScatteredOut` object containing the scattering transform results, `index` specifies which example in the batch to plot, 
`origSig` is the original input signal, `saveTo` specifies the file path to save the GIF, and `fps` sets the frames per second for the GIF animation. 
If `saveTo` is provided, the GIF is saved to that file path. The default `fps` is set to 2 frames per second. 
"""
function gifFirstLayer1D(origLoc, origSig; fps=2, title="First Layer", saveTo=nothing, index=1)
    anim = Animation()
    for j = 1:size(origLoc[1])[end-1]
        plotFirstLayer1DSingleWavelet(j, origLoc, origSig; index=index)
        frame(anim)
    end
    filepath = isnothing(saveTo) ? "tmp.gif" : saveTo
    mkpath(dirname(filepath))
    return gif(anim, filepath, fps=fps)
end

"""
    plotFirstLayer1DAll(origLoc, origSig; title="First Layer", saveTo=nothing, index=1, cline=:darkrainbow)
Function that plots all first layer gradient wavelets for a specific example signal `index` across space, along with the original signal. 
It also includes heatmaps of the gradient wavelets in both the spatial and frequency domains. 
The variable `index` specifies which example in the batch to plot, `origLoc` is the `ScatteredOut` object 
containing the scattering transform results, `origSig` is the original input signal, and `saveTo` is the file path to save the plot.
"""
function plotFirstLayer1DAll(origLoc, origSig; title="First Layer", saveTo=nothing, index=1, cline=:darkrainbow)
    space = plot(origLoc[1][:, :, index], line_z=(1:size(origLoc[1], 2))', xlim=(0, length(origLoc[1][:, 1, index])+1), 
        legend=false, colorbar=true, color=cline, title="$title Gradient Wavelets")
    org = plot(origSig[:,:,index], legend=false, color=:red, title="Original Signal", xlim=(0, length(origSig[:,:,index])+1))
    ∇h = heatmap(origLoc[1][:, 1:end, index]', xlabel="space",
        ylabel="wavelet index", title="First Layer gradients")
    ∇̂h = heatmap(log.(abs.(rfft(origLoc[1][:, 1:end, index], 1)) .^ 2)', xlabel="frequency",
        ylabel="wavelet index", title="Log-power Frequency Domain")
    l = Plots.@layout [a; b{0.1h}; [b c]]
    plt = plot(space, org, ∇h, ∇̂h, layout=l, size=(1280, 720), margin=5Plots.mm)
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
    plotFirstLayer1D(stw, St; title="First Layer", saveTo=nothing, index=1)
Function that creates a heatmap of the first layer scattering transform results at a specified example index. 
The variable `stw` is the scattered output, `St` is the scattering transform object, `saveTo` is the file 
path to save the plot, and `index` specifies which example in the batch to plot.
"""
function plotFirstLayer1D(stw, St; title="First Layer", saveTo=nothing, index=1)
    f1, f2, f3 = getMeanFreq(St) # the mean frequencies for the wavelets in each layer. 
    plt = heatmap(1:size(stw[1], 1), f1[1:end-1], stw[1][:, :, index]', 
            xlabel="time index", ylabel="Frequency (Hz)", margin=5Plots.mm,
            color=:viridis, title="$title", size=(1280, 720))
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end


meanWave(wave) = sum(real.(range(0, stop=1, length=size(wave, 1)) .* wave), dims=1) ./ sum(real.(wave), dims=1)

# As far as I can tell, this function is not used elsewhere. 
"""
    plotSecondLayer1DOld(loc, origLoc, wave1, wave2, original=false, subsamSz=(128,85,))
"""
function plotSecondLayer1DOld(loc, origLoc, wave1, wave2, original=false, subsamSz=(128, 85,), c=:thermal, lastDiagFreq=true)
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
    plotSecondLayer1DSpecificPath(stw, St, firstLayerWaveletIndex, secondLayerWaveletIndex, original; saveTo=nothing, index=1)
`stw` is the scattered output, `St` is the scattering transform object, `firstLayerWaveletIndex` and `secondLayerWaveletIndex` specify the path to plot, `original` is the original signal, and `index` specifies 
which example in the batch to plot. This function creates a plot showing the original signal and the scattering result for the specified path. It also displays the mean frequencies associated with the 
selected wavelets. Finally, it displays the log-power norm of the second layer signal for the specified path. This value is used elsewhere to create heatmaps of the second layer scattering results. 
"""
function plotSecondLayer1DSpecificPath(stw, St, firstLayerWaveletIndex, secondLayerWaveletIndex, original; saveTo=nothing, index=1)
    # Plot of original signal. 
    org = plot(original[:,:,index], legend=false, color=:red, title="Original Signal", xlabel="time (samples)", ylabel="amplitude", xlims=(0, length(original[:,:,index])+1))
    f1, f2, f3 = getMeanFreq(St)
    
    # Plot the signal for a specific path. 
    signalLayer1Freq = f1[firstLayerWaveletIndex]; signalLayer2Freq = f2[secondLayerWaveletIndex]
    plotsTitle = "Path: First Layer - wavelet $firstLayerWaveletIndex, Second Layer - wavelet $secondLayerWaveletIndex\n" *
                "First Layer Freq = $(round(signalLayer1Freq, sigdigits=3)) Hz | " * 
                "Second Layer Freq = $(round(signalLayer2Freq, sigdigits=3)) Hz"
    titlePlot = plot(title=plotsTitle, grid=false, showaxis=false, xticks=nothing, yticks=nothing, bottom_margin=-5Plots.px, titlefontsize=11)
    
    path_spatial = stw[2][:, secondLayerWaveletIndex, firstLayerWaveletIndex, index]
    ∇h = plot(path_spatial, xlabel="time (samples)", ylabel="amplitude", title="Second Layer Plot", legend=false, linewidth=1.5, frame=:box, fill=0, fillalpha=0.5, 
              xlims=(0, length(path_spatial)), ylims=(0, maximum(path_spatial)*1.01))

    secondLayerNorm = log10.(norm(stw[2][:, secondLayerWaveletIndex, firstLayerWaveletIndex, index]))
    normPlot = plot(title="Second Layer Signal norm (log-power) = $(round(secondLayerNorm, sigdigits=4))",
                    grid=false, showaxis=false, xticks=nothing, yticks=nothing, titlefontsize=11, framestyle=:none)
    l = Plots.@layout [a{0.4h}; title{0.05h}; c; d{0.05h}]
    plt = plot(org, titlePlot, ∇h, normPlot, layout=l, margin=4Plots.mm, size=(1080,720))
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
    gifSecondLayer1DSubset(stw, St, firstLayerWavelets, secondLayerWavelets, original; fps=2, saveTo=nothing, index=1)
Create a GIF visualizing the second layer scattering results for specified subsets of wavelets from the first and second layers. The variables `firstLayerWavelets` and `secondLayerWavelets` are arrays 
containing the indices of the wavelets to be visualized from the first and second layers, respectively. For example, to visualize all the wavelets from the first layer with respect to a specific 
wavelet from the second layer, you can set `firstLayerWavelets = 1:size(stw[1], 2)` and `secondLayerWavelets = k`, where `k` is the index of the desired second layer wavelet. Once again, the `index` 
parameter specifies which example in the batch to plot. It defaults to the first example in the batch. If `saveTo` is not provided, the GIF is saved to "tmp.gif".
"""
function gifSecondLayer1DSubset(stw, St, firstLayerWavelets, secondLayerWavelets, original; fps=2, saveTo=nothing, index=1)
    anim = Animation()
    for j in firstLayerWavelets, k in secondLayerWavelets
        plt = plotSecondLayer1DSpecificPath(stw, St, j, k, original; saveTo=nothing, index=index)
        frame(anim, plt)
    end
    mkpath(dirname(saveTo))
    filepath = isnothing(saveTo) ? "tmp.gif" : saveTo
    return gif(anim, filepath, fps=fps)
end

"""
    plotSecondLayer1DFixAndVary(stw, St, firstLayerWavelets, secondLayerWavelets; fps=2, saveTo=nothing, index=1)
Create a GIF visualizing slices of the second layer scattering results by fixing one layer's wavelet and varying the other layer's wavelet. 
The variables `firstLayerWavelets` and `secondLayerWavelets` are arrays containing the indices of the wavelets to be visualized from the first and second layers, respectively. 
If `firstLayerWavelets` contains only one index, the function fixes that wavelet and varies the second layer wavelets, and vice versa.
If `saveTo` is not provided, the GIF is saved to "tmp.gif".
"""
function plotSecondLayer1DFixAndVary(stw, St, firstLayerWavelets, secondLayerWavelets; fps=2, saveTo=nothing, index=1)
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
            plt = heatmap(1:size(toPlot, 1), f1[1:end-1], toPlot', title="Second Layer Wavelet $jj, Frequency=$(round(f2[jj], sigdigits=4))Hz", 
                    xlabel="time (samples)", ylabel="First Layer Frequency (Hz)", c=cgrad(:viridis, scale=:exp), margin=5Plots.mm, size=(1080,720))
            frame(anim, plt)
        end
    else
        # Fixed second layer, vary first layer.
        for jj in firstLayerWavelets
            if jj > size(stw[2], 3)
                continue
            end
            toPlot = stw[2][:, :, jj, index]
            plt = heatmap(1:size(toPlot, 1), f2[1:end-1], toPlot', title="First Layer Wavelet $jj, Frequency=$(round(f1[jj], sigdigits=4))Hz", 
                    xlabel="time (samples)", ylabel="Second Layer Frequency (Hz)", c=cgrad(:viridis, scale=:exp), margin=5Plots.mm, size=(1080,720))
            frame(anim, plt)
        end
    end
    mkpath(dirname(saveTo))
    filepath = isnothing(saveTo) ? "tmp.gif" : saveTo
    return gif(anim, filepath, fps=fps)
end

"""
    plotSecondLayer1D(stw, St; saveTo=nothing, index=1, title="Second Layer results", xVals=-1, yVals=-1, logPower=true, toHeat=nothing, c=cgrad(:viridis, [0,.9]), threshold=0, linePalette=:greys, minLog=NaN, kwargs...)
TODO fix the similarity of these names.
xVals and yVals give the spacing of the grid, as it doesn't seem to be done
correctly by default. xVals gives the distance from the left and the right
as a tuple, while yVals gives the distance from the top and the bottom,
also as a tuple. Default values are `xVals = (.037, .852), yVals = (.056, .939)`, or if you have no title, use `xVals = (.0105, .882), yVals = (.056, .939)`
If you have no colorbar, set `xVals = (.0015, .997), yVals = (.002, .992)`
In the case that arbitrary space has been introduced, if you have a title, use `xVals = (.037, .852), yVals = (.056, .939)`, or if you have no title, use `xVals = (.0105, .882), yVals = (.056, .939)`
"""
function plotSecondLayer1D(stw::ScatteredOut, St; saveTo=nothing, index=1, title="Second Layer results", xVals=-1, yVals=-1, logPower=true, toHeat=nothing, c=cgrad(:viridis, [0,.9]), threshold=0, linePalette=:greys, minLog=NaN, kwargs...)
    secondLayerRes = stw[2]
    if ndims(secondLayerRes) > 3
        return plotSecondLayer1D(secondLayerRes[:, :, :, index], St; saveTo=saveTo, kwargs...)
    else
        return plotSecondLayer1D(secondLayerRes, St; saveTo=saveTo, kwargs...)
    end
end

function plotSecondLayer1D(stw, St; saveTo=nothing, title="Second Layer results", xVals=-1, yVals=-1, logPower=true, toHeat=nothing, c=cgrad(:viridis, [0, 0.9]), threshold=0, freqsigdigits=3, linePalette=:greys, minLog=NaN, subClims=(Inf, -Inf), δt=1000, firstFreqSpacing=nothing, secondFreqSpacing=nothing, transp=true, labelRot=30, xlabel=nothing, ylabel=nothing, frameTypes=:box, miniFillAlpha=0.5, kwargs...)
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
    # Substitute given x and y labels if needed
    if isnothing(xlabel)
        xlabel = "Layer $(xInd) frequency (Hz)"
    end
    if isnothing(ylabel)
        ylabel = "Layer $(yInd) frequency (Hz)"
    end

    if title == ""
        plt = heatmap(toHeat; yticks=yTicksFreq, xticks=xTicksFreq, tick_direction=:out, rotation=labelRot,
            xlabel=xlabel, ylabel=ylabel, left_margin=5Plots.mm, bottom_margin=3Plots.mm, c=c, clims=(bottom, top), size=(1280,1080), kwargs...)
    else
        plt = heatmap(toHeat; yticks=yTicksFreq, xticks=xTicksFreq, tick_direction=:out, rotation=labelRot,
            title=title, xlabel=xlabel, ylabel=ylabel, left_margin=5Plots.mm, bottom_margin=3Plots.mm, c=c, clims=(bottom, top), size=(1280,1080), kwargs...)
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
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
    jointPlot1DOld(thingToPlot, thingName, cSymbol, St; saveTo=nothing, sharedColorScaling=:exp, targetExample=1, δt=1000, freqigdigits=3, sharedColorbar=false, extraPlot=nothing, allPositive=false, logPower=false)
Create a joint plot visualizing the zeroth, first, and second layer scattering results for a specified example. The variable `thingToPlot` is a tuple containing the scattering results for the 
zeroth, first, and second layers, `thingName` is the title for the plot, `cSymbol` specifies the color gradient to use, and `St` is the scattering transform object. The function allows for 
various customization options, including shared color scaling, target example selection, frequency digit rounding, and additional plotting options. In this implementation, `targetExample` is the 
same as `index` in other functions, specifying which example in the batch to plot.
"""
function jointPlot1DOld(thingToPlot, thingName, cSymbol, St; saveTo=nothing, sharedColorScaling=:exp, targetExample=1, δt=1000, freqigdigits=3, sharedColorbar=false, extraPlot=nothing, allPositive=false, logPower=false, kwargs...)
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

    p2 = plotSecondLayer(thingToPlot[2][:, :, :, targetExample], St; title="Second Layer", toHeat=toHeat, logPower=logPower, c=c, clims=climssecond, subClims=climssecond, cbar=false, xVals=(0.000, 0.993), yVals=(0.0, 0.994), xlabel="", transp=true, kwargs...)
    freqs = getMeanFreq(St, δt)
    freqs = map(x -> round.(x, sigdigits=freqigdigits), freqs)
    p1 = heatmap(firstLay, c=c, title="First Layer", clims=climsfirst, cbar=false, yticks=((1:size(firstLay, 1)), ""), xticks=((1:size(firstLay, 2)), ""), bottom_margin=0Plots.px)
    p0 = heatmap(spaceLocs, 1:1, zeroLay, c=c, xlabel="Zeroth Layer", clims=climszero, cbar=false, yticks=nothing, top_margin=0Plots.px, bottom_margin=5Plots.px)
    colorbarOnly = scatter([0, 0], [0, 1], zcolor=[0, 3], clims=climssecond, xlims=(1, 1.1), xshowaxis=false, yshowaxis=false, label="", c=c, grid=false, framestyle=:none)
    if extraPlot == nothing
        # extraPlot = scatter([0,0], [0,1], legend=false, grid=false, x="Layer 2 frequency", foreground_color_subplot=:white, top_margin=-10Plots.px, showaxis=false, yticks=nothing)
        extraPlot = plot(xlabel="Layer 2 frequency (Hz)", grid=false, xticks=(1:5, ""), showaxis=false, yticks=nothing, bottom_margin=0Plots.px, top_margin=3Plots.px)
        # extraPlot = heatmap(zeroLay, c=c, xlabel="location\nZeroth Layer", clims=climszero, cbar=false, yticks=nothing, top_margin=-10Plots.px, bottom_margin=10Plots.px)
    end
    titlePlot = plot(title=thingName, grid=false, showaxis=false, xticks=nothing, yticks=nothing, bottom_margin=0Plots.px)
    lay = Plots.@layout [o{0.00001h}; [[a b; c{0.1h} d{0.1h}] b{0.04w}]]
    plt = plot(titlePlot, p2, p1, extraPlot, p0, colorbarOnly, layout=lay, size=(1500,1000), margin=6Plots.mm, top_margin=2Plots.mm)
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end


"""
    jointPlot1D(thingToPlot, thingName, cSymbol, St, origSig; saveTo=nothing, sharedColorScaling=:exp, targetExample=1, δt=1000, freqigdigits=3, sharedColorbar=false, allPositive=false, logPower=false)
Create a joint plot visualizing the zeroth, first, and second layer scattering results for a specified example. The variable `thingToPlot` is a tuple containing the scattering results for the 
zeroth, first, and second layers, `thingName` is the title for the plot, `cSymbol` specifies the color gradient to use, and `St` is the scattering transform object. The function allows for 
various customization options, including shared color scaling, target example selection, frequency digit rounding, and additional plotting options. In this implementation, `targetExample` is the 
same as `index` in other functions, specifying which example in the batch to plot. Finally, `origSig` is the original input signal to be displayed alongside the scattering results.
"""
function jointPlot1D(thingToPlot, thingName, cSymbol, St, origSig; saveTo=nothing, sharedColorScaling=:exp, targetExample=1, δt=1000, freqigdigits=3, sharedColorbar=false, allPositive=false, logPower=false, kwargs...)
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

    p2 = plotSecondLayer1D(thingToPlot[2][:, :, :, targetExample], St; title="Second Layer", toHeat=toHeat, logPower=logPower, c=c, clims=climssecond, subClims=climssecond, cbar=false, xVals=(0.000, 0.993), yVals=(0.0, 0.994), transp=true, kwargs...)
    freqs = getMeanFreq(St, δt)
    freqs = map(x -> round.(x, sigdigits=freqigdigits), freqs)
    p1 = heatmap(firstLay, c=c, title="First Layer", clims=climsfirst, cbar=false, yticks=((1:size(firstLay, 1)), ""), xticks=((1:size(firstLay, 2)), ""), bottom_margin=0Plots.px, left_margin=0Plots.mm)
    p0 = heatmap(spaceLocs, 1:1, zeroLay, c=c, xlabel="Zeroth Layer", clims=climszero, cbar=false, yticks=nothing, top_margin=6Plots.mm, xguidefontsize=12, left_margin=0Plots.mm)
    colorbarOnly = scatter([0, 0], [0, 1], zcolor=[0, 3], clims=climssecond, xlims=(1, 1.1), xshowaxis=false, yshowaxis=false, label="", c=c, grid=false, framestyle=:none)
    originalSignalPlot = plot(origSig[:,1,targetExample], xlabel="Original Signal", legend=false, xlim=(0, length(origSig[:,1,targetExample])+1), color=:blue, top_margin=6Plots.mm, xguidefontsize=12)
    titlePlot = plot(title=thingName, grid=false, showaxis=false, xticks=nothing, yticks=nothing, top_margin=0Plots.px, bottom_margin=0Plots.px)
    lay = Plots.@layout [o{0.01h}; [[a b; c{0.1h} d{0.1h}] cb{0.04w}]]
    plt = plot(titlePlot, p2, p1, originalSignalPlot, p0, colorbarOnly, layout=lay, size=(1920,1080), margin=5Plots.mm, left_margin=10Plots.mm)
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end



"""
    plotOriginalSignal2D(origSig; color=:grays, saveTo=nothing, index=1)
Function that plots the original signal `origSig` and saves it to a desired location.  
"""
function plotOriginalSignal2D(origSig; color=:grays, saveTo=nothing, index=1)
    plt = heatmap(origSig[:,:,1,index], title="Original Signal", legend=false, axis=false, color=color, colorbar=false, margin=5Plots.mm, size=(720,480))
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
    plotZerothLayer2D(sf; color=:grays, saveTo=nothing, index=1)
Function that plots the zeroth layer of the scattering transform at a specified example index. 
"""
function plotZerothLayer2D(sf; color=:grays, saveTo=nothing, index=1)
    plt = heatmap(sf[0][:,:,1,index], title="Zeroth Layer", legend=false, axis=false, color=color, margin=5Plots.mm, size=(720,480))
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
    plotFirstLayer2DSingleWavelet(j, sf, origSig; color=:grays, saveTo=nothing, index=1)
The variable `j` specifies which wavelet results to plot from the first layer, `index` specifies which example in the batch to plot, 
`sf` is the `ScatteredOut` object containing the scattering transform results, and `origSig` is the original input signal. 
"""
function plotFirstLayer2DSingleWavelet(j, sf, origSig; color=:grays, saveTo=nothing, index=1)
    space = heatmap(sf[1][:,:,j,index], title="First Layer - Gradient Wavelet $j", legend=false, axis=false, color=color)
    orig = heatmap(origSig[:,:,1,index], title="Original Signal", legend=false, axis=false, color=color)
    l = Plots.@layout [a b]
    plt = plot(space, orig, layout=l, size=(1280, 720), margin=5Plots.mm)
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
        visualizeFirstLayer2D(sf; color=:grays, index=1) 
Helper function that creates a grid of heatmaps visualizing all the first layer gradient wavelets across space for a specified example index. 
The variable `sf` is the `ScatteredOut` object containing the scattering transform results, `index` specifies which example in the batch to plot, and 
`color` sets the color gradient for the heatmaps. The function organizes the heatmaps in a grid format, where each row corresponds to a different 
scale of the wavelets, and each column corresponds to a different channel (e.g., real and imaginary parts).
"""
function visualizeFirstLayer2D(sf; color=:grays, index=1, clims=nothing, colorScaling=false)
    nchannel = 3
    scale = Int(size(sf[1], 3) / nchannel)

    clims = isnothing(clims) ? (minimum(sf[1][:,:,:,index]), maximum(sf[1][:,:,:,index])) : clims

    plots = []
    for scal = 1:scale
        o1 = sf[1][:,:,1 + 3*(scal-1), index]
        o2 = sf[1][:,:,2 + 3*(scal-1), index]
        o3 = sf[1][:,:,3 + 3*(scal-1), index]
        if colorScaling
            if scal == 1
                push!(plots, heatmap(o1, color=color, colorbar=false, axis=false, legend=false, ylabel=string(scal), title="1", clims=clims))
                push!(plots, heatmap(o2, color=color, colorbar=false, axis=false, legend=false, title="i", clims=clims))
                push!(plots, heatmap(o3, color=color, colorbar=false, axis=false, legend=false, title="j", clims=clims))
            else
                push!(plots, heatmap(o1, color=color, colorbar=false, axis=false, legend=false, ylabel=string(scal), clims=clims))
                push!(plots, heatmap(o2, color=color, colorbar=false, axis=false, legend=false, clims=clims))
                push!(plots, heatmap(o3, color=color, colorbar=false, axis=false, legend=false, clims=clims))
            end
        else
            if scal == 1
                push!(plots, heatmap(o1, color=color, colorbar=false, axis=false, legend=false, ylabel=string(scal), title="1"))
                push!(plots, heatmap(o2, color=color, colorbar=false, axis=false, legend=false, title="i"))
                push!(plots, heatmap(o3, color=color, colorbar=false, axis=false, legend=false, title="j"))
            else
                push!(plots, heatmap(o1, color=color, colorbar=false, axis=false, legend=false, ylabel=string(scal)))
                push!(plots, heatmap(o2, color=color, colorbar=false, axis=false, legend=false))
                push!(plots, heatmap(o3, color=color, colorbar=false, axis=false, legend=false))
            end
        end
    end
    return plot(plots..., layout=(scale, nchannel), size=(540, 1080))
end

"""
    plotFirstLayer2D(sf; color=:grays, saveTo=nothing, index=1)
Function that creates a heatmap of the first layer scattering transform results at a specified example index. 
The variable `sf` is the scattered output, `saveTo` is the file path to save the plot, and `index` specifies which example in the batch to plot.
The user can also specify the color gradient for the heatmap using the `color` argument. 
"""
function plotFirstLayer2D(sf; color=:grays, saveTo=nothing, index=1)
    nchannel = 3
    scale = Int(size(sf[1], 3) / nchannel)
    clims = (minimum(sf[1][:,:,:,index]), maximum(sf[1][:,:,:,index]))

    plots = []
    for scal = 1:scale
        o1 = sf[1][:,:,1 + 3*(scal-1), index]
        o2 = sf[1][:,:,2 + 3*(scal-1), index]
        o3 = sf[1][:,:,3 + 3*(scal-1), index]
        if scal == 1
            push!(plots, heatmap(o1, color=color, colorbar=false, axis=false, legend=false, ylabel=string(scal), title="1", clims=clims))
            push!(plots, heatmap(o2, color=color, colorbar=false, axis=false, legend=false, title="i", clims=clims))
            push!(plots, heatmap(o3, color=color, colorbar=false, axis=false, legend=false, title="j", clims=clims))
        else
            push!(plots, heatmap(o1, color=color, colorbar=false, axis=false, legend=false, ylabel=string(scal), clims=clims))
            push!(plots, heatmap(o2, color=color, colorbar=false, axis=false, legend=false, clims=clims))
            push!(plots, heatmap(o3, color=color, colorbar=false, axis=false, legend=false, clims=clims))
        end
    end
    plt = plot(plots..., layout=(scale, nchannel), plot_title="First Layer", size=(1080, 960), margin=5Plots.mm)

    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
    plotFirstLayer2DAll(sf, origSig; saveTo=nothing, index=1, color=:grays)
Function that plots all first layer gradient wavelets for a specific example signal `index` across space, along with the original signal. 
The variable `index` specifies which example in the batch to plot, `sf` is the `ScatteredOut` object 
containing the scattering transform results, `origSig` is the original input signal, and `saveTo` is the file path to save the plot.
"""
function plotFirstLayer2DAll(sf, origSig; saveTo=nothing, index=1, color=:grays, colorScaling=false)
    global_min = min(minimum(sf[1][:,:,:,index]), minimum(origSig[:,:,1,index]))
    global_max = max(maximum(sf[1][:,:,:,index]), maximum(origSig[:,:,1,index]))
    clims = (global_min, global_max)

    if colorScaling
        space = visualizeFirstLayer2D(sf; color=color, index=index, clims=clims, colorScaling=false)
        orig = heatmap(origSig[:,:,1,index], title="Original Signal", legend=false, axis=false, color=color, colorbar=false, clims=clims)
    else
        space = visualizeFirstLayer2D(sf; color=color, index=index)
        orig = heatmap(origSig[:,:,1,index], title="Original Signal", legend=false, axis=false, color=color, colorbar=false)
    end

    l = Plots.@layout [a b{1.0w}]
    plt = plot(space, orig, layout=l, size=(1280, 960), margin=5Plots.mm)
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end


"""
    plotSecondLayer2DSingleWavelet(j, sf, origSig; color=:grays, saveTo=nothing, index=1)
The variable `j` specifies which wavelet results to plot from the second layer, `index` specifies which example in the batch to plot, 
`sf` is the `ScatteredOut` object containing the scattering transform results, and `origSig` is the original input signal. 
"""
function plotSecondLayer2DSingleWavelet(firstLayerWaveletIndex, secondLayerWaveletIndex, sf, origSig; color=:grays, saveTo=nothing, index=1)
    space = heatmap(sf[2][:,:,secondLayerWaveletIndex,firstLayerWaveletIndex,index], title="Second Layer - Path $firstLayerWaveletIndex -> $secondLayerWaveletIndex", legend=false, axis=false, color=color)
    orig = heatmap(origSig[:,:,1,index], title="Original Signal", legend=false, axis=false, color=color)
    l = Plots.@layout [a b]
    plt = plot(space, orig, layout=l, size=(1280, 720), margin=5Plots.mm)
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end

"""
    visualizeSecondLayer2D(sf; color=:grays, index=1)
Function that creates a grid of heatmaps visualizing all the second layer gradient wavelets across space for a specified example index. 
The variable `sf` is the `ScatteredOut` object containing the scattering transform results, `index` specifies which example in the batch to plot, and `color` sets the color gradient for the heatmaps. 
The function organizes the heatmaps in a grid format, where each row corresponds to a different scale of the second layer wavelets, and each column corresponds to a different channel (e.g., real and imaginary parts). 
"""
function visualizeSecondLayer2D(sf; color=:grays, index=1, clims=nothing)
    nchannel = 3
    n_layer1 = size(sf[2], 4)
    n_layer2 = size(sf[2], 3)
    scale_layer1 = Int(n_layer1 / nchannel)
    scale_layer2 = Int(n_layer2 / nchannel)

    clims = isnothing(clims) ? (minimum(sf[2][:,:,:,:,index]), maximum(sf[2][:,:,:,:,index])) : clims
    plots = []

    col_titles = ["(1,:)", "(i,:)", "(j,:)"]
    row_titles = ["(:,1)", "(:,i)", "(:,j)"]

    for scal2 = 1:scale_layer2
        for k = 1:nchannel
            for scal1 = 1:scale_layer1
                for j = 1:nchannel
                    o = sf[2][:, :, k + nchannel*(scal2-1), j + nchannel*(scal1-1), index]

                    is_top_row    = (scal2 == 1) && (k == 1)
                    is_left_col   = (scal1 == 1) && (j == 1)
                    is_right_col  = (scal1 == scale_layer1) && (j == nchannel)
                    is_bottom_row = (scal2 == scale_layer2) && (k == nchannel)

                    title_str  = is_top_row  ? col_titles[j]   : ""
                    ylabel_str = is_left_col ? string(scal2)   : ""
                    xlabel_str = is_bottom_row ? string(scal1) : ""

                    hm = heatmap(o, color=color, axis=false, legend=false,
                                 title=title_str, ylabel=ylabel_str, xlabel=xlabel_str, clims=clims)
                    if is_right_col
                        annotate!(hm, [(size(o,2)*1.15, size(o,1)/2, 
                                       Plots.text(row_titles[k], 8, :left, :black))])
                    end
                    push!(plots, hm)
                end
            end
        end
    end

    total_rows = n_layer2
    total_cols = n_layer1
    return plot(plots..., layout=(total_rows, total_cols), margin=2Plots.mm, size=(2000, 2200))
end

"""
    plotSecondLayer2D(sf; color=:grays, saveTo=nothing, index=1)
Function that creates a grid of heatmaps visualizing all the second layer scattering transform results 
at a specified example index. The variable `sf` is the `ScatteredOut` object, `saveTo` is the file path 
to save the plot, `index` specifies which example in the batch to plot, and `color` sets the color gradient.
The grid rows correspond to second layer scales/channels and columns to first layer scales/channels.
"""
function plotSecondLayer2D(sf; color=:grays, saveTo=nothing, index=1)
    nchannel = 3
    n_layer1 = size(sf[2], 4)
    n_layer2 = size(sf[2], 3)
    scale_layer1 = Int(n_layer1 / nchannel)
    scale_layer2 = Int(n_layer2 / nchannel)

    clims = (minimum(sf[2][:,:,:,:,index]), maximum(sf[2][:,:,:,:,index]))
    plots = []

    col_titles = ["(1,:)", "(i,:)", "(j,:)"]
    row_titles = ["(:,1)", "(:,i)", "(:,j)"]

    for scal2 = 1:scale_layer2
        for k = 1:nchannel
            for scal1 = 1:scale_layer1
                for j = 1:nchannel
                    o = sf[2][:, :, k + nchannel*(scal2-1), j + nchannel*(scal1-1), index]

                    is_top_row    = (scal2 == 1) && (k == 1)
                    is_left_col   = (scal1 == 1) && (j == 1)
                    is_right_col  = (scal1 == scale_layer1) && (j == nchannel)
                    is_bottom_row = (scal2 == scale_layer2) && (k == nchannel)

                    title_str  = is_top_row  ? col_titles[j]   : ""
                    ylabel_str = is_left_col ? string(scal2)   : ""
                    xlabel_str = is_bottom_row ? string(scal1) : ""

                    hm = heatmap(o, color=color, axis=false, legend=false,
                                 title=title_str, ylabel=ylabel_str, xlabel=xlabel_str, clims=clims)
                    if is_right_col
                        annotate!(hm, [(size(o,2)*1.15, size(o,1)/2, 
                                       Plots.text(row_titles[k], 8, :left, :black))])
                    end
                    push!(plots, hm)
                end
            end
        end
    end

    total_rows = n_layer2
    total_cols = n_layer1
    plt = plot(plots..., layout=(total_rows, total_cols), plot_title="Second Layer", 
               size=(2000, 2200), margin=2Plots.mm, dpi=150)
    if !isnothing(saveTo)
        mkpath(dirname(saveTo))
        savefig(plt, saveTo)
    end
    return plt
end