@testset "Bayes.jl" begin

    Random.seed!(42)

    # HPDregions with a data vector
    data = randn(2000)
    hpd, cov = HPDregions(data, 0.95)
    @test size(hpd, 2) == 2
    @test cov >= 0.95
    @test hpd[1,1] < hpd[1,2]

    # HPDregions with an Nx1 matrix (triggers the size(data,2)==1 reshape branch)
    hpd2, cov2 = HPDregions(reshape(randn(1000), 1000, 1), 0.90)
    @test size(hpd2, 2) == 2

    # HPDregions with 2D data should error
    @test_throws ErrorException HPDregions(randn(10, 2), 0.95)

    # HPDregions with a distribution (interior support — no boundary branch triggered)
    hpd3, cov3 = HPDregions(Normal(0, 1), 0.95)
    @test size(hpd3, 2) == 2
    @test cov3 >= 0.95
    @test abs(hpd3[1,1] + hpd3[1,2]) < 0.1  # approximately symmetric around 0

    # HPDregions with Beta(0.5, 2): pdf → ∞ at left boundary → triggers min boundary branch
    hpd4, cov4 = HPDregions(Beta(0.5, 2.0), 0.90)
    @test size(hpd4, 2) == 2
    @test cov4 >= 0.90

    # HPDregions with Beta(2, 0.5): pdf → ∞ at right boundary → triggers max boundary branch
    hpd5, cov5 = HPDregions(Beta(2.0, 0.5), 0.90)
    @test size(hpd5, 2) == 2
    @test cov5 >= 0.90

end
