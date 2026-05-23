function slice_sample_bounded(logπ, θ_curr; lower=-1.0, upper=1.0)
    log_u = logπ(θ_curr) + log(rand())
    L, R = lower, upper
    while true
        θ_prop = L + rand() * (R - L)
        logπ(θ_prop) > log_u && return θ_prop
        θ_prop < θ_curr ? (L = θ_prop) : (R = θ_prop)
    end
end

# test - sample from a truncated normal distribution
# using Distributions.jl for comparison
using Distributions, Plots

α = 3.0
β = 2.0
lower = 0.5
upper = 10.0
nsim = 50000
dist = Truncated(Gamma(α, β), lower, upper)
logπ(x) = logpdf(dist, x)
samples = zeros(nsim)
samples[1] = (lower + upper) / 2
for i in 2:nsim
    samples[i] = slice_sample_bounded(logπ, samples[i-1]; lower=lower, upper=upper)
end
println("Sample mean: ", mean(samples))
println("Theoretical mean: ", mean(rand(dist, nsim)))
println("Sample std: ", std(samples))
println("Theoretical std: ", std(rand(dist, nsim)))

histogram(samples; bins=50, density=true, label="Slice Sampling", alpha=0.5,
    normalize=true)
x = 0:0.01:(upper+2)
plot!(x, pdf.(dist, x); label="Truncated Gamma PDF", lw=2)