"""

  given a path p and a flattened scattering transform flat, return the transform of that row.
"""
function findp(flat::Vector{T}, p::Vector{S}, layers::layeredTransform) where {T <: Number, S<:Integer}

end


"""
  p = computePath(layers::layeredTransform, layerM::Int64, λ::Int64; stType::String="full"))

compute the path p that corresponds to location λ in the stType of transform
"""
function computePath(layers::layeredTransform, layered1D::scattered1D, layerM::Int64, λ::Int64; stType::String="full")
  nScalesLayers = [numScales(layers.shears[i], layered1D.data[i]) for i=1:layerM]
  p = zeros(layerM)
  if stType=="full"
    # start at the outermost level and work in
    for i=1:layerM
      p[i] = div(tmpλ, prod(nScalesLayers[i+1:end]))+1
      tmpλ -= (p[i]-1)*prod(nScalesLayers[i+1:end])
    end
  elseif stType=="decreasing"

  end
end


"""
    numChildren = numChildren(keeper::Array{Int64}, layers::layeredTransform, nScalesLayers::Array{Int64})

given a keeper, determine the number of children it has.
"""
function numChildren(keeperOriginal::Array{Int64}, scalingFactors::Array{Float64}, nScalesLayers::Array{Int64})
  keeperOriginal = [keeperOriginal; 1]
  keeper = copy(keeperOriginal)
  m = length(keeper)
  if m==0
    return 1 # not sure why you bothered
  elseif m==1
    numThisLayer = 1
  else
    numThisLayer = 0
  end
  while true
    if keeperOriginal==[12, 12, 1]
    end
    # we're done if either if we've hit the maximum number of scales, or if at least one of the old indices is larger than it used to be. In the first case, count it, in the second case, don't
    if keeper[m]==nScalesLayers[m]
      return numThisLayer+=1
    elseif reduce((a,b)->a|b, [keeper[i]>keeperOriginal[i] for i=1:m-1])
      if keeperOriginal==[12, 12, 1]
      end
      return numThisLayer
    end
    numThisLayer += 1
    isLast, keeper = incrementKeeper(keeper, m, scalingFactors, nScalesLayers)
    if isLast
      break
    end
  end
  numThisLayer
end
numChildren(keeperOriginal::Array{Int64}, layers::layeredTransform, nScalesLayers::Array{Int64}) = numChildren(keeperOriginal::Array{Int64}, [shear.scalingFactor for shear in layers.shears], nScalesLayers::Array{Int64})

"""
numInLayer(m::Int64, layers::layeredTransform, nScalesLayers::Array{Int64})

A stupidly straightforward way of counting the number of decreasing paths in a given layer.
"""
function numInLayer(m::Int64, scalingFactors::Array{Float64}, nScalesLayers::Array{Int64})
  keeper = [1 for i=1:m]
  if m==0
    return 1 # not sure why you bothered
  else
    numThisLayer = 1
  end
  while true
    isLast, keeper = incrementKeeper(keeper, m, scalingFactors, nScalesLayers)
      if isLast
      break
    end
    numThisLayer+=1
  end
  numThisLayer
end
numInLayer(m::Int64, layers::layeredTransform, nScalesLayers::Array{Int64}) = numInLayer(m, [shear.scalingFactor for shear in layers.shears], nScalesLayers)
"""
    (isLast, keeper) = incrementKeeper(keeper::Array{Float64}, m::Int64, resolutions::Array{Float64})
keeper is a tuple where the ith entry is the scale used at layer i. This function is designed so that it only returns decreasing entries (including effects from the scales per octave). It does so in order from smallest scale in all layers, to largest in all layers, incrementing the deepest layers first. isLast is a boolean which is true if this is the last entry.
"""
function incrementKeeper(keeper::Array{Int64}, m::Int64, scalingFactors::Array{Float64,1}, nScalesLayers::Array{Int64})
  keeper[m]+=1
  if m==1
    # if we've cycled all possible values, we're done, so return an empty keeper
    if keeper[1] > nScalesLayers[1]
      return (true, Array{Int64}(0))
    end
  else
    # if we've cycled this layer as much as possible, we move on
    if keeper[m] > nScalesLayers[m] || floor(keeper[m-1]./scalingFactors[m-1]) < ceil(keeper[m]./scalingFactors[m])
      keeper[m]=1
      return incrementKeeper(keeper, m-1, scalingFactors, nScalesLayers)
    end
  end
  return (false, keeper)
end
incrementKeeper(keeper::Array{Int64}, m::Int64, layers::layeredTransform, nScalesLayers::Array{Int64}) = incrementKeeper(keeper, m, [shear.scalingFactor for shear in layers.shears], nScalesLayers::Array{Int64})