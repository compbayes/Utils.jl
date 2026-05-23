function slice_sample_bounded(logπ, θ_curr; lower=-1.0, upper=1.0)
    log_u = logπ(θ_curr) + log(rand())
    L, R = lower, upper
    while true
        θ_prop = L + rand() * (R - L)
        logπ(θ_prop) > log_u && return θ_prop
        θ_prop < θ_curr ? (L = θ_prop) : (R = θ_prop)
    end
end