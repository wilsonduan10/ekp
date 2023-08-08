# Imports
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

using Downloads
using DelimitedFiles
using NCDatasets

using CLIMAParameters
const CP = CLIMAParameters
FT = Float64

import RootSolvers
const RS = RootSolvers

import SurfaceFluxes as SF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFP
using StaticArrays: SVector

# We include some helper files. The first is to set up the parameters for surface\_conditions, and
# the second is to plot our results.
include("../helper/setup_parameter_set.jl")
include("../helper/graph.jl")

mkpath(joinpath(@__DIR__, "../images/SHEBA_profiles"))

include("5level_data.jl")
u_star_data = vec(mean(u_star_data, dims=1))

heatmap(u_data,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="Time", ylabel="Level",)
png("images/SHEBA_profiles/u_data")

heatmap(z_data,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="Time", ylabel="Level",)
png("images/SHEBA_profiles/z_data")

heatmap(q_data,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="Time", ylabel="Level",)
png("images/SHEBA_profiles/qt_data")

heatmap(T_data,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="Time", ylabel="Level",)
png("images/SHEBA_profiles/temp_data")

heatmap(shf_data,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="Time", ylabel="Level",)
png("images/SHEBA_profiles/shf_data")

plot(u_star_data, seriestype=:scatter, ms=1.5, c=:red, markerstrokecolor=:red)
png("images/SHEBA_profiles/u_star_data")

plot(T_sfc_data, seriestype=:scatter, ms=1.5, c=:red, markerstrokecolor=:red)
png("images/SHEBA_profiles/surface_temp_data")

plot(p_data, seriestype=:scatter, ms=1.5, c=:red, markerstrokecolor=:red)
png("images/SHEBA_profiles/surface_pressure_data")