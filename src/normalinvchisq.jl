import Statistics: mean

import Distributions:
    rand,
    pdf,
    logpdf,
    params



# NORMAL INV Gamma

# Used "Conjugate Bayesian analysis of the Gaussian distribution" by Murphy as
# a reference.  Note that there were some typos in that document so the code
# here may not correspond exactly.

struct NormalInverseGamma{T<:Real} <: ContinuousUnivariateDistribution
    mu::T
    v0::T     # scales variance of Normal
    shape::T  
    scale::T

    function NormalInverseGamma{T}(mu::T, v0::T, sh::T, r::T) where T<:Real
    	v0 > zero(v0) && sh > zero(sh) && r > zero(r) || error("Both shape and scale must be positive")
    	new{T}(T(mu), T(v0), T(sh), T(r))
    end
end

function NormalInverseGamma(mu::Real, v0::Real, sh::Real, r::Real)
    T = promote_type(typeof(mu), typeof(v0), typeof(sh), typeof(r))
    return NormalInverseGamma{T}(T(mu),T(v0),T(sh),T(r))
end

mu(d::NormalInverseGamma) = d.mu
v0(d::NormalInverseGamma) = d.v0
shape(d::NormalInverseGamma) = d.shape
scale(d::NormalInverseGamma) = d.scale
rate(d::NormalInverseGamma) = 1. / d.scale

insupport(::Type{NormalInverseGamma}, x::T, sig2::T) where T<:Real = 
    isfinite(x) && zero(sig2) <= sig2 < Inf 

# Probably should guard agains dividing by and taking the log of 0.

function pdf(d::NormalInverseGamma, x::T, sig2::T) where T<:Real
    Zinv = d.scale.^d.shape / gamma(d.shape) / sqrt(d.v0 * 2.0*pi)
    return Zinv * 1.0/(sqrt(sig2)*sig2.^(d.shape+1.0)) * exp(-d.scale/sig2 - 0.5/(sig2*d.v0)*(x-d.mu).^2)
end

function logpdf(d::NormalInverseGamma, x::T, sig2::T) where T<:Real
    lZinv = d.shape*log(d.scale) - lgamma(d.shape) - 0.5*(log(d.v0) + log(2pi))
    return lZinv - 0.5*log(sig2) - (d.shape+1.)*log(sig2) - d.scale/sig2 - 0.5/(sig2*d.v0)*(x-d.mu).^2
end

function mode(d::NormalInverseGamma)
    mu = d.mu
    sig2 = d.scale / (d.shape + 1.5)
    return mu, sig2
end

function mean(d::NormalInverseGamma)
    mu = d.mu
    sig2 = d.shape > 1.0 ? d.scale / (d.shape - 1.0) : Inf
    return mu, sig2
end

function rand(d::NormalInverseGamma)
    # Guard against invalid precisions
    sig2 = rand(InverseGamma(d.shape, d.scale))
    if sig2 <= zero(Float64)
        sig2 = eps(Float64)
    end
    mu = rand(Normal(d.mu, sqrt(sig2*d.v0)))
    return mu, sig2
end


### NORMAL INV CHI SQUARE
"""
    NormalInverseChisq(μ, σ2, κ, ν)

A Normal-χ^-2 distribution is a conjugate prior for a Normal distribution with
unknown mean and variance.  It has parameters:

* μ: expected mean
* σ2 > 0: expected variance
* κ ≥ 0: mean confidence
* ν ≥ 0: variance confidence

The parameters have a natural interpretation when used as a prior for a Normal
distribution with unknown mean and variance: μ and σ2 are the expected mean and
variance, while κ and ν are the respective degrees of confidence (expressed in
"pseudocounts").  When interpretable parameters are important, this makes it a
slightly more convenient parametrization of the conjugate prior.

Equivalent to a `NormalInverseGamma` distribution with parameters:

* m0 = μ
* v0 = 1/κ
* shape = ν/2
* scale = νσ2/2

Based on Murphy "Conjugate Bayesian analysis of the Gaussian distribution".
"""
struct NormalInverseChisq{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ2::T
    κ::T
    ν::T

    function NormalInverseChisq{T}(μ::T, σ2::T, κ::T, ν::T) where T<:Real
        if ν < 0 || κ < 0 || σ2 ≤ 0
            throw(ArgumentError("Variance and confidence (κ and ν) must all be positive"))
        end
        new{T}(μ, σ2, κ, ν)
    end
end

NormalInverseChisq() = NormalInverseChisq{Float64}(0.0, 1.0, 0.0, 0.0)

function NormalInverseChisq(μ::Real, σ2::Real, κ::Real, ν::Real)
    T = promote_type(typeof(μ), typeof(σ2), typeof(κ), typeof(ν))
    NormalInverseChisq{T}(T(μ), T(σ2), T(κ), T(ν))
end

Base.convert(::Type{NormalInverseGamma}, d::NormalInverseChisq) =
    NormalInverseGamma(d.μ, 1/d.κ, d.ν/2, d.ν*d.σ2/2)

Base.convert(::Type{NormalInverseChisq}, d::NormalInverseGamma) =
    NormalInverseChisq(d.mu, d.scale/d.shape, 1/d.v0, d.shape*2)

insupport(::Type{NormalInverseChisq}, μ::T, σ2::T) where T<:Real =
    isfinite(μ) && zero(σ2) <= σ2 < Inf

params(d::NormalInverseChisq) = d.μ, d.σ2, d.κ, d.ν

function pdf(d::NormalInverseChisq, μ::T, σ2::T) where T<:Real
    Zinv = sqrt(d.κ / 2pi) / gamma(d.ν*0.5) * (d.ν * d.σ2 / 2)^(d.ν*0.5)
    Zinv * σ2^(-(d.ν+3)*0.5) * exp( (d.ν*d.σ2 + d.κ*(d.μ - μ)^2) / (-2 * σ2))
end

function logpdf(d::NormalInverseChisq, μ::T, σ2::T) where T<:Real
    logZinv = (log(d.κ) - log(2pi))*0.5 - lgamma(d.ν*0.5) + (log(d.ν) + log(d.σ2) - log(2)) * (d.ν/2)
    logZinv + log(σ2)*(-(d.ν+3)*0.5) + (d.ν*d.σ2 + d.κ*(d.μ - μ)^2) / (-2 * σ2)
end

function mean(d::NormalInverseChisq)
    μ = d.μ
    σ2 = d.ν/(d.ν-2)*d.σ2
    return μ, σ2
end

function mode(d::NormalInverseChisq)
    μ = d.μ
    σ2 = d.ν*d.σ2/(d.ν + 3)
    return μ, σ2
end

rand(d::NormalInverseChisq) = rand(convert(NormalInverseGamma, d))

