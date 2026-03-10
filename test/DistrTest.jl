@testset "Distr.jl" begin

    @test Distributions.var(TDist(3, 2, 5)) == 2^2*(5/(5-2))
    @test Distributions.mean(ScaledInverseChiSq(10, 5)) == 5*(10/(10-2))

    # GaussianCopula: pdf equals MvNormal pdf when margins are Gaussian
    μ₁ = 10; μ₂ = 0; σ₁ = 1; σ₂ = 3; σ12 = -1.9
    ρ_corr = σ12/(σ₁*σ₂)
    Ω = PDMat([1 ρ_corr; ρ_corr 1])
    MvN = MvNormal([μ₁; μ₂], [σ₁^2 σ12; σ12 σ₂^2])
    Random.seed!(1)
    x = rand(MvN)
    GC = GaussianCopula(Ω, [Normal(μ₁, σ₁), Normal(μ₂, σ₂)])
    @test pdf(GC, x) ≈ pdf(MvN, x)
    @test logpdf(GC, x) ≈ log(pdf(GC, x))

    # GaussianCopula: length
    @test length(GC) == 2

    # GaussianCopula: singleton distribution constructor
    GC_single = GaussianCopula(Ω, Normal(0.0, 1.0))
    @test length(GC_single) == 2
    @test GC_single.f[1] == GC_single.f[2]

    # GaussianCopula: rand single sample (non-diagonal → correlated branch)
    x_rand = rand(GC)
    @test length(x_rand) == 2

    # GaussianCopula: rand n samples
    X_rand = rand(GC, 5)
    @test size(X_rand) == (2, 5)

    # GaussianCopula: diagonal CorrMat (isdiag branch in rand)
    GC_diag = GaussianCopula(PDMat(Matrix(1.0*I(2))), [Normal(0.0, 1.0), Exponential(1.0)])
    @test (rand(GC_diag); true)  # just verify it executes

    # GaussianCopula: vector-only constructor (identity CorrMat)
    GC_vec = GaussianCopula(UnivariateDistribution[Normal(0.0, 1.0), Normal(1.0, 2.0)])
    @test length(GC_vec) == 2

    # SimDirProcess
    Random.seed!(99)
    θ_sim, π_sim = SimDirProcess(Normal(), 5.0, 0.001)
    @test length(θ_sim) == length(π_sim)
    @test issorted(θ_sim)
    @test sum(π_sim) < 1.0

    # PGDistOneParam
    d_pg = PGDistOneParam(1, 10)
    p_pg = pdf(d_pg, 1.0)
    @test p_pg > 0
    @test logpdf(d_pg, 1.0) ≈ log(p_pg)

    # --- NormalInverseGamma (not exported) ---
    d_nig = Utils.NormalInverseGamma(0.0, 1.0, 2.0, 1.0)

    @test Utils.insupport(Utils.NormalInverseGamma, 0.0, 1.0)
    @test !Utils.insupport(Utils.NormalInverseGamma, 0.0, -1.0)

    p_nig = pdf(d_nig, 0.0, 1.0)
    @test p_nig > 0
    @test logpdf(d_nig, 0.0, 1.0) ≈ log(p_nig)

    μ_nig, σ2_nig = mean(d_nig)
    @test μ_nig == 0.0
    @test σ2_nig > 0

    # shape ≤ 1: variance mean returns Inf
    d_nig_low = Utils.NormalInverseGamma(0.0, 1.0, 0.5, 1.0)
    _, σ2_inf = mean(d_nig_low)
    @test σ2_inf == Inf

    μ_mode_nig, σ2_mode_nig = Utils.mode(d_nig)
    @test μ_mode_nig == 0.0
    @test σ2_mode_nig > 0

    μ_r_nig, σ2_r_nig = rand(d_nig)
    @test isfinite(μ_r_nig)
    @test σ2_r_nig > 0

    # Invalid NIG constructor
    @test_throws ErrorException Utils.NormalInverseGamma(0.0, -1.0, 2.0, 1.0)

    # --- NormalInverseChisq ---
    d_nic0 = NormalInverseChisq()
    @test d_nic0.μ == 0.0

    d_nic = NormalInverseChisq(1.0, 2.0, 3.0, 4.0)
    @test params(d_nic) == (1.0, 2.0, 3.0, 4.0)

    @test Utils.insupport(NormalInverseChisq, 1.0, 2.0)
    @test !Utils.insupport(NormalInverseChisq, 1.0, -1.0)

    p_nic = pdf(d_nic, 1.0, 2.0)
    @test p_nic > 0
    @test logpdf(d_nic, 1.0, 2.0) ≈ log(p_nic)

    μ_nic, σ2_nic = mean(d_nic)
    @test μ_nic == 1.0
    @test σ2_nic > 0

    μ_nic_mode, σ2_nic_mode = Utils.mode(d_nic)
    @test μ_nic_mode == 1.0
    @test σ2_nic_mode > 0

    μ_nic_r, σ2_nic_r = rand(d_nic)
    @test isfinite(μ_nic_r)
    @test σ2_nic_r > 0

    # NIC ↔ NIG roundtrip conversion
    d_as_nig = convert(Utils.NormalInverseGamma, d_nic)
    d_back = convert(NormalInverseChisq, d_as_nig)
    @test d_back.μ ≈ d_nic.μ
    @test d_back.σ2 ≈ d_nic.σ2

    # Invalid NIC constructors
    @test_throws ArgumentError NormalInverseChisq(0.0, -1.0, 1.0, 1.0)  # σ2 ≤ 0
    @test_throws ArgumentError NormalInverseChisq(0.0, 1.0, -1.0, 1.0)  # κ < 0
    @test_throws ArgumentError NormalInverseChisq(0.0, 1.0, 1.0, -1.0)  # ν < 0

    # NegativeBinomial2
    μ = 0.5; ϕ = 2.3;
    d = NegativeBinomial2(μ, ϕ)
    @test mean(d) ≈ μ
    @test var(d) ≈ μ*(1 + μ/ϕ)
end
