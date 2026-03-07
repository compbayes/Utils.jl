@testset "PlotUtils.jl" begin

    gr()  # use GR backend (headless)

    # PlotSettings.jl: mvcolors palette
    @test length(mvcolors) == 12

    # plotFcnGrid: error branch (no StepRangeLen in xGrid → neither 1D/2D/3D)
    f_test = (x, args...) -> sum(x .^ 2)
    @test_throws ErrorException plotFcnGrid(f_test, [1.0, 2.0], ["x1", "x2"])

    # plotFcnGrid: 1D branch (one StepRangeLen, one fixed Number)
    p1d = plotFcnGrid(f_test, [range(-1.0, 1.0, length=20)], ["x"])
    @test !isnothing(p1d)

    # plotFcnGrid: 1D branch with a fixed argument alongside the grid
    p1d_fixed = plotFcnGrid(f_test, [0.5, range(-1.0, 1.0, length=20)], ["a", "b"])
    @test !isnothing(p1d_fixed)

    # plot_braces!: horizontal + upward (defaults: up=true, horizontal=true)
    plot(-3:0.1:3, sin.(-3:0.1:3))
    @test (plot_braces!(0.0, 1.0, 2.0, 0.1; lw=1); true)

    # plot_braces!: horizontal + downward (up=false)
    @test (plot_braces!(0.0, 0.5, 2.0, 0.1, false; lw=1); true)

    # plot_braces!: vertical (horizontal=false)
    @test (plot_braces!(0.0, 0.5, 2.0, 0.1, true, false; lw=1); true)

end
