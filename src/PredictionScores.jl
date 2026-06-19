# Dawid–Sebastiani score for evaluating probabilistic forecast based on a multivariate normal predictive distribution (Dawid & Sebastiani, 1999).


"""   
    dawid_sebastiani_score(samples::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    
Compute the Dawid–Sebastiani score 

DSS = log(det(Σ)) + (y - μ)' * Σ^(-1) * (y - μ)

for evaluating probabilistic forecasts based on a (approximate) multivariate normal predictive distribution (Dawid & Sebastiani, 1999).
- `samples`: n x p matrix of n draws from the predictive distribution, each row a p-vector.
- `y`: the p-vector observation to be scored.
"""
function dawid_sebastiani_score(samples::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    n, p = size(samples)
    d = length(y)
    p == d || throw(ArgumentError("length of `y` must match the dimension of the draws"))
    μ = mean(samples, dims=1)[:]
    Σ = cov(samples, dims=1)

    C = cholesky(Symmetric(Σ))
    z = C.L \ (y - μ)
    return sum(abs2, z) + logdet(C)

end

"""
    crps(samples::AbstractVector{<:Real}, y::Real)

Unbiased empirical CRPS for one observation `y`, given `n` equally-weighted
draws `samples` from the predictive distribution.

Computed in O(n log n) via sorting rather than the naive O(n^2) double sum.
"""
function crps(samples::AbstractVector{<:Real}, y::Real)

    n = length(samples)
    term1 = mean(abs(x - y) for x in samples)
    s = sort(samples)
    acc = 0.0
    @inbounds for i in 1:n
        acc += (2i - n - 1) * s[i]
    end
    term2 = acc / n^2   # = (1/2) * (1/n^2) * sum_i sum_j |s_i - s_j|

    return term1 - term2
end


"""
    energy_score(samples::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                 draws_dim::Int = 1)

Energy score: the multivariate generalization of CRPS (Gneiting & Raftery,
2007). For a p-vector target `y` and n joint draws of a p-vector:

    ES(F, y) = E||X - y|| - (1/2) E||X - X'||,   X, X' ~ F iid, ||.|| Euclidean

"""
function energy_score(samples::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
    draws_dim::Int=1)

    draws_dim in (1, 2) || throw(ArgumentError("draws_dim must be 1 or 2"))
    X = draws_dim == 1 ? samples : permutedims(samples)   # n x p, rows = draws
    n, p = size(X)
    p == length(y) || throw(ArgumentError("length of `y` must match the dimension of the draws"))

    D = pairwise(Euclidean(), X, dims=1)   # n x n distance matrix

    term1 = 0.0
    @inbounds for i in 1:n
        d2 = 0.0
        for k in 1:p
            d2 += (X[i, k] - y[k])^2
        end
        term1 += sqrt(d2)
    end
    term1 /= n

    term2 = sum(D) / (2 * n^2)

    return term1 - term2
end


"""
    variogram_score(samples::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                     p::Real = 0.5,
                     weights::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                     draws_dim::Int = 1)

Variogram score (Scheuerer & Hamill, 2015) -- a multivariate proper scoring
rule that is invariant to location but sensitive to correct correlatiin. 
`p` is the order; Scheuerer & Hamill recommend p = 0.5. 
`weights` defaults to all ones (every pair weighted equally); pass a d x d matrix to standardize across components with very different scales, e.g. `W = 1 ./ (sd * sd')` using each component's predictive standard deviation.

`draws_dim = 1` (default): `samples` is n x d, each row a draw.
`draws_dim = 2`           : `samples` is d x n, each column a draw.
"""
function variogram_score(samples::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
    p::Real=0.5,
    weights::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    draws_dim::Int=1)
    draws_dim in (1, 2) || throw(ArgumentError("draws_dim must be 1 or 2"))
    X = draws_dim == 1 ? samples : permutedims(samples)   # n x d, rows = draws
    n, d = size(X)
    d == length(y) || throw(ArgumentError("length of `y` must match the dimension of the draws"))

    W = weights === nothing ? ones(d, d) : weights
    size(W) == (d, d) || throw(ArgumentError("`weights` must be d x d"))

    # Predicted pairwise E|X_i - X_j|^p, estimated from the draws.
    Epow = Matrix{Float64}(undef, d, d)
    @inbounds for i in 1:d, j in 1:d
        acc = 0.0
        for k in 1:n
            acc += abs(X[k, i] - X[k, j])^p
        end
        Epow[i, j] = acc / n
    end

    vs = 0.0
    @inbounds for i in 1:d, j in 1:d
        obs = abs(y[i] - y[j])^p
        vs += W[i, j] * (obs - Epow[i, j])^2
    end

    return vs
end
