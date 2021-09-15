# Exploring the conditional posterior of τ
# Set up full conditional for τ with gradient and Hessian functions using AD.
function logCondPostτ(τ, y, X, β)
    p = exp.(X*β)./(1 .+ exp.(X*β))
    logLik = sum( logpdf.(NegativeBinomial.(exp(τ), p), y) )
    logPrior = logpdf(Normal(μₜ,σₜ), τ)
    return logLik + logPrior
end
πₜargs = [y,X,β]
ℓπₜ(τ, πₜargs...) = logCondPostτ(τ, πₜargs...)
∇ₜ(τ,πₜargs...) = ForwardDiff.derivative(τ -> ℓπₜ(τ, πₜargs...), τ)
Hₜ(τ,πₜargs...) = ForwardDiff.derivative(τ -> ∇ₜ(τ,πₜargs...), τ)


# Compare with RWM sampling. 
negBinReg3 = DensityModel(τ -> logCondPostτ(τ, πₜargs...))
nBurnin = Int(0.5*nIter)
postSamp = DataFrame(sample(negBinReg3, RWMH(Normal(0,0.01)), nIter+nBurnin; param_names=["τ"]))
mean(postSamp[(nBurnin+1):end,1])
std(postSamp[(nBurnin+1):end,1])
density(postSamp[(nBurnin+1):end,1])
kdeOpt = kde(postSamp[(nBurnin+1):end,1])
plot(kdeOpt.x, kdeOpt.density)
plot!(kdeOpt.x, pdf(Normal(mean(postSamp[(nBurnin+1):end,1]), √(-1/Hₜ(1.05,πₜargs...))), kdeOpt.x), color = colors[3])
#######################