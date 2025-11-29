
""" 
    HPDregions(data::AbstractArray, coverage) 

Compute the highest posterior density (HPD) regions based on a kernel density estimate of the `data`. 

`coverage` ∈ (0,1) is the probability mass to be included in the HPD region.

# Examples
```julia-repl
julia> hpdregion, actualCoverage = HPDregions(randn(100), 0.95)
```
""" 
function HPDregions(data::AbstractArray, coverage)

    if size(data,2) == 1 
        data = data[:]
    else
        error("Data must be a vector")
    end
    kdeObject = kde(data) 
    binSize = step(kdeObject.x)
    sortidx = sortperm(kdeObject.density, rev=true) # descending
    xSort = kdeObject.x[sortidx] 
    densSort = kdeObject.density[sortidx]
    
    finalpointin = findfirst(cumsum(densSort*binSize) .>= coverage)
    actualCoverage = cumsum(densSort*binSize)[finalpointin]
    hpdPoints = sort(xSort[1:finalpointin])
    breakpoints = findall(diff(hpdPoints) .> 1.9*binSize)
    nIntervals = length(breakpoints) + 1
    breakpoints = [0;breakpoints;length(hpdPoints)]

    hpd = zeros(nIntervals,2)
    for j = 1:nIntervals
        hpd[j,:] = [ hpdPoints[breakpoints[j]+1] hpdPoints[breakpoints[j+1]] ]
    end

    return hpd, actualCoverage
end

""" 
    HPDregions(d::UnivariateDistribution, coverage) 

Compute the highest posterior density (HPD) regions for the distribution `d`. 

`coverage` ∈ (0,1) is the probability mass to be included in the HPD region.

# Examples
```julia-repl
julia> hpdregion, actualCoverage = HPDregions(Normal(0,1), 0.95)
```
""" 
function HPDregions(d::UnivariateDistribution, coverage)

    min, max = quantile.(d,[0.001,0.999])
    # Check if end of support has higher density
    if pdf(d, min) < pdf(d, minimum(Distributions.support(d)))
        min = minimum(Distributions.support(d))
    end
    if pdf(d, max) < pdf(d, maximum(Distributions.support(d)))
        max = maximum(Distributions.support(d))
    end
    xGrid = range(min, max, length = 1000)
    binSize = step(xGrid)
    dens = pdf.(d, xGrid)
    sortidx = sortperm(dens, rev=true) # descending
    xSort = xGrid[sortidx] 
    densSort = dens[sortidx]
    
    finalpointin = findfirst(cumsum(densSort*binSize) .>= coverage)
    actualCoverage = cumsum(densSort*binSize)[finalpointin]
    hpdPoints = sort(xSort[1:finalpointin])
    breakpoints = findall(diff(hpdPoints) .> 1.9*binSize)
    nIntervals = length(breakpoints) + 1
    breakpoints = [0;breakpoints;length(hpdPoints)]

    hpd = zeros(nIntervals,2)
    for j = 1:nIntervals
        hpd[j,:] = [ hpdPoints[breakpoints[j]+1] hpdPoints[breakpoints[j+1]] ]
    end

    return hpd, actualCoverage 
end