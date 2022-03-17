@testset "Distr.jl" begin
    
    @test var(TDist(3, 2, 5)) == 2^2*(5/(5-2))

    @test mean(ScaledInverseChiSq(10, 5)) == 5*(10/(10-2))

    zdist = ZDist(1/2,1/2)
    @test Utils.cdf.(zdist,Utils.quantile.(zdist, 0.1:0.1:0.9)) ≈ 0.1:0.1:0.9

    zdist = ZDist(3/2,3/2)
    @test Utils.cdf.(zdist,Utils.quantile.(zdist, 0.1:0.1:0.9)) ≈ 0.1:0.1:0.9
end