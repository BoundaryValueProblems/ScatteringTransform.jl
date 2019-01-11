########################################################
########################################################
# PLOTTING
########################################################
########################################################

"""
    plot = comparePathsChildren(thinResult::Array{Float64,N}, path::pathType, layers::layeredTransform; outputSubsample=(-1,3), scale::Symbol=:log) where N
compares the children of the given path across the first dimensions; don't try to plot too many entries simultaneously. scale gives the color scaling; should be either :log or :linear. path should be a pathType representing the parent.
If pathType is ommitted, it will make a plot for each and save those plots in a directory with the given name
"""
function comparePathsChildren(thinResult, path::pathType, layers::layeredTransform; outputSubsample=(-1,3), names=["" for i=1:size(thinResult,1)], colorScheme=:curl, scale::Symbol=:log, title="",titleSize=12) where N
    k, n, q, dataSizes, outputSizes, resultingSize = ScatteringTransform.calculateThinStSizes(layers, outputSubsample, (size(thinResult)[1:end-1],layers.n))
    thinStDims = pathToThinIndex(path, layers, outputSubsample)
    toPlot = thinResult[axes(thinResult)[1:end-1]..., thinStDims]
    toPlot = reshape(toPlot,(size(thinResult)[1:end-1]...,floor(Int,length(thinStDims)/resultingSize[path.m+1]), resultingSize[path.m+1]))
    ranges = (minimum(toPlot), maximum(toPlot))
    colorMeMine = cgrad(colorScheme,scale=scale)
    if title==""
        return plot([heatmap(toPlot[i,:,:], xticks=1:size(toPlot)[end], clims=ranges, fillcolor=colorMeMine,title=names[i]) for i=1:size(toPlot,1)]...)
    else
        return plot([heatmap(toPlot[i,:,:], xticks=1:size(toPlot)[end], clims=ranges, fillcolor=colorMeMine, title=names[i]) for i=1:size(toPlot, 1)]..., plot(annotations = (.5,.5,text(title, :center, titleSize)), axis=false, xticks=[], yticks=[]))
    end
end
# TODO: only works for depth of 2 at the moment
function comparePathsChildren(thinResult, layers::layeredTransform; outputSubsample=(-1,3), saveDirectory="", names=["" for i=1:size(thinResult,1)], colorScheme=:curl, scale::Symbol=:log, title="",titleSize=12) where N
    mkpath(saveDirectory)
    listOfPaths = [pathType(0,[])]
    n = sizes(bspline, layers.subsampling, layers.n)
    q = [numScales(layers.shears[i],n[i]) for i=1:layers.m+1]
    push!(listOfPaths, [pathType(1,i) for i=1:q[1]-1]...)
    for path in listOfPaths
        pathIndex = (path.m==0) ? "" : path.Idxs[1]
        savefig(comparePathsChildren(thinResult, path, layers; outputSubsample=outputSubsample, names=names, colorScheme=colorScheme, scale=scale, title="Layer $(path.m+1) Path $(pathIndex)", titleSize=titleSize),joinpath(saveDirectory,"Layer$(path.m+1)Path$(pathIndex).pdf"))
    end
end
