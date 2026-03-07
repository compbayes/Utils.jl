@testset "Utils.jl" begin

    # --- LinAlgMisc.jl ---

    @test invvech([11,21,22], 2) == [11 21;21 22]
    @test invvech([11,21,31,22,32,33], 3) == [11 21 31;21 22 32;31 32 33]

    # fillupper = false: only lower triangle populated
    lower = invvech([11,21,31,22,32,33], 3; fillupper=false)
    @test lower[2,1] == 21
    @test lower[1,2] == 0

    @test invvech_byrow([11,21,22], 2) == [11 21;21 22]
    @test invvech_byrow([11,21,22,31,32,33], 3) == [11 21 31;21 22 32;31 32 33]

    lower_br = invvech_byrow([11,21,22,31,32,33], 3; fillupper=false)
    @test lower_br[2,1] == 21
    @test lower_br[1,2] == 0

    @test CovMatEquiCorr([1.0,1.0], [0.0,0.0], [1,1]) == I(2)

    Random.seed!(7)
    σₓ = rand(5)
    ρ = rand(2)
    pBlock = [2,3]
    CovMat = CovMatEquiCorr(σₓ, ρ, pBlock)
    @test CovMat[1,3] == 0
    @test CovMat[1,2]/(√CovMat[1,1]*√CovMat[2,2]) ≈ ρ[1]

    ρComputed, σComputed = Cov2Corr(CovMat)
    @test ρComputed[1,2] ≈ ρ[1]
    @test ρComputed[4,5] ≈ ρ[2]
    @test σComputed[1] ≈ σₓ[1]

    # --- Misc.jl ---

    A = [1 2 3; 4 5 6; 7 8 9]
    @test find_min_matrix(A, 3)[3] == CartesianIndex(1, 3)
    @test find_max_matrix(A, 3)[1] == CartesianIndex(3, 3)

    # subscript (not exported)
    @test Utils.subscript(0) == "₀"
    @test Utils.subscript(45) == "₄₅"
    @test_throws ErrorException Utils.subscript(-1)

    # pad_digits (not exported)
    padded = Utils.pad_digits([1.21, 13.3, 123.456])
    @test padded == ["1.210", "13.300", "123.456"]

    # optimalPlotLayout — all explicit branches
    @test optimalPlotLayout(1)  == (1, 1)
    @test optimalPlotLayout(2)  == (1, 2)
    @test optimalPlotLayout(3)  == (2, 2)
    @test optimalPlotLayout(4)  == (2, 2)
    @test optimalPlotLayout(5)  == (2, 3)
    @test optimalPlotLayout(6)  == (2, 3)
    @test optimalPlotLayout(7)  == (3, 3)
    @test optimalPlotLayout(8)  == (3, 3)
    @test optimalPlotLayout(9)  == (3, 3)
    @test optimalPlotLayout(10) == (3, 4)
    @test optimalPlotLayout(11) == (3, 4)
    @test optimalPlotLayout(12) == (3, 4)
    @test optimalPlotLayout(13) == (4, 4)
    @test optimalPlotLayout(14) == (4, 4)
    @test optimalPlotLayout(15) == (4, 4)
    @test optimalPlotLayout(16) == (4, 4)
    # fallthrough (> 16): ceil(sqrt(n)) × ceil(sqrt(n))
    let n = 25
        r, c = optimalPlotLayout(n)
        @test r == ceil(sqrt(n))
        @test c == ceil(sqrt(n))
    end

    # quantile_multidim
    B = reshape(Float64.(1:12), 3, 4)  # 3×4, column-major
    q = quantile_multidim(B, 0.5; dims=1)   # median of each column → 1×4
    @test size(q) == (1, 4)
    @test q[1,1] ≈ 2.0   # median of [1,2,3]

end
