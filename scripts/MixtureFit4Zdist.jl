# Script for approximating the Z(α,β) distribution by a mixture.

using Distributions, Distances, Optim

figFolder = "/home/mv/Dropbox/Julia/dev/SpecLocalStat/scripts/figs/"

function distScaleMix2Target(θ, targetDist, compDist, xGrid, distFunc)
    K = length(compDist)
    σ = exp.(θ[1:K]) 
    w = exp.(θ[(K+1):end]) ./ sum(exp.(θ[(K+1):end]))
    dMix = MixtureModel(σ .* compDist, w)
    return evaluate(distFunc, pdf.(targetDist, xGrid), pdf.(dMix, xGrid) )
end

function distMix2Target(θ, targetDist, compDist, xGrid, distFunc)
    K = length(compDist)
    μ = θ[1:K]
    σ = exp.(θ[(K+1):2K]) 
    w = exp.(θ[(2K+1):end]) ./ sum(exp.(θ[(2K+1):end]))
    dMix = MixtureModel(σ .* compDist .+ μ, w)
    return evaluate(distFunc, pdf.(targetDist, xGrid), pdf.(dMix, xGrid) )
end

xGrid = -50:0.05:50
distFunc = Euclidean()
distFunc = KLDivergence()
targetDist = ZDist(1/2,1/2)
symmetric = pdf(targetDist, -1) ≈ pdf(targetDist, 1)
maxK = 4
selQuants = [10.0^j for j ∈ -4:0.01:-2]
quants = zeros(length(selQuants),maxK)
p = []
for K = 1:maxK
    compDist = [TDist(10) for _ ∈ 1:K];
    if symmetric
        θ₀ = [zeros(K);repeat([1/K],K)];
        optRes = maximize(θ -> -distScaleMix2Target(θ, targetDist, compDist, xGrid, distFunc), θ₀);
        θopt = optRes.res.minimizer;
        μ = zeros(K);
        σ = exp.(θopt[1:K]);
        w = exp.(θopt[(K+1):end]) ./ sum(exp.(θopt[(K+1):end]));
        dMix = MixtureModel(σ .* compDist, w)
    else
        θ₀ = [zeros(2K);repeat([1/K],K)];
        optRes = maximize(θ -> -distMix2Target(θ, targetDist, compDist, xGrid, distFunc), θ₀);
        θopt = optRes.res.minimizer;
        μ = θopt[1:K]
        σ = exp.(θopt[(K+1):2K]);
        w = exp.(θopt[(2K+1):end]) ./ sum(exp.(θopt[(2K+1):end]));
        dMix = MixtureModel(σ .* compDist .+ μ, w)
    end

    ptmp = plot(xGrid, pdf.(targetDist, xGrid), label = "Target", 
        title = "K = $K, distance = $(round(optRes.res.minimum, digits = 4))")
    plot!(ptmp, xGrid, pdf.(dMix, xGrid), label = "Mixture")
    push!(p, ptmp)
    quants[:,K] = quantile.(dMix, selQuants)
end
plot(size = (1000, 800), p..., layout = (2,2))
savefig(figFolder*"Densities.pdf")

q = []
for K = 1:maxK
    qtmp = plot(selQuants, quantile.(targetDist, selQuants), color = :black, label = "Target", title = L"K = %$K", ylab = "quantiles", xlab = "tail probability")
    plot!(qtmp, selQuants, quants[:,K], label = "Mixture", legend = :bottomright)
    push!(q, qtmp)
end
plot(size = (1000, 800), q..., layout = (2,2))
savefig(figFolder*"Quantiles.pdf")
