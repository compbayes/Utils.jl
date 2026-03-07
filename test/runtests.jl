ENV["GKSwstype"] = "100"  # headless GR backend for plotting tests

using Utils, Distributions, LinearAlgebra, PDMats
using Test, Random
using Plots

include("UtilsTest.jl")
include("DistrTest.jl")
include("BayesTest.jl")
include("PlotUtilsTest.jl")
