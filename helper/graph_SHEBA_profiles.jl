using LinearAlgebra, Random
using Distributions, Plots
using Downloads
using NCDatasets
import SurfaceFluxes as SF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP

ENV["GKSwstype"] = "nul"
include("../helper/setup_parameter_set.jl")
mkpath(joinpath(@__DIR__, "../data")) # create data folder if not exists
mkpath(joinpath(@__DIR__, "../images"))
mkpath(joinpath(@__DIR__, "../images/SHEBA_profiles"))

# get data from data folder
EC_tend_filepath = "data/EC_tend.nc"
ECMWF_filepath = "data/ECMWF.nc"
surf_obs_filepath = "data/surf_obs.nc"

ECMWF_data = NCDataset(ECMWF_filepath)
EC_data = NCDataset(EC_tend_filepath)
surf_obs_data = NCDataset(surf_obs_filepath)

# I index all data from ECMWF and EC_tend starting from 169 in order to align it with the data
# from surf_obs_data, so that all data start on October 29, 1997
time_data = Array(ECMWF_data["yymmddhh"])[169:end] # (8112, )
time_data2 = Array(surf_obs_data["Jdd"]) # (8112, )
time_data3 = Array(EC_data["yymmddhh"])[169:end] # (8112, )
u_star_data = Array(surf_obs_data["ustar"]) # (8112, )
u_data = Array(EC_data["u"])[:, 169:end]
v_data = Array(EC_data["v"])[:, 169:end]
qv = Array(ECMWF_data["qv"])[:, 169:end]
ql = Array(ECMWF_data["ql"])[:, 169:end]
qi = Array(ECMWF_data["qi"])[:, 169:end]
surface_temp_data = Array(surf_obs_data["T_sfc"]) # (8112, )
surface_pressure_data = Array(ECMWF_data["psurf"])[169:end]
temp_data = Array(ECMWF_data["T"])[:, 169:end]
lhf_data = Array(surf_obs_data["hl"]) # (8112, )
shf_data = Array(surf_obs_data["hs"]) # (8112, )
z0m_data = Array(ECMWF_data["surf-roughness-length"])[169:end] # (8112, )
z0b_data = Array(ECMWF_data["surface-roughness-length-heat"])[169:end] # (8112, )
p_data = Array(ECMWF_data["p"])[:, 169:end]
qt_data = qv .+ ql .+ qi # (31, 8112)

Z, T = size(u_data)

for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
end

heatmap(u_data,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="Time", ylabel="Level",)
png("images/SHEBA_profiles/u_data")

heatmap(qt_data,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="Time", ylabel="Level",)
png("images/SHEBA_profiles/qt_data")

heatmap(temp_data,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="Time", ylabel="Level",)
png("images/SHEBA_profiles/temp_data")

heatmap(p_data,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="Time", ylabel="Level",)
png("images/SHEBA_profiles/pressure_data")

plot(u_star_data, seriestype=:scatter, ms=1.5, c=:red, markerstrokecolor=:red)
png("images/SHEBA_profiles/u_star_data")

plot(surface_temp_data, seriestype=:scatter, ms=1.5, c=:red, markerstrokecolor=:red)
png("images/SHEBA_profiles/surface_temp_data")

plot(surface_pressure_data)
png("images/SHEBA_profiles/surface_pressure_data")

plot(lhf_data, seriestype=:scatter, ms=1.5, c=:yellow, markerstrokecolor=:yellow)
png("images/SHEBA_profiles/lhf_data")

plot(shf_data, seriestype=:scatter, ms=1.5, c=:red, markerstrokecolor=:red)
png("images/SHEBA_profiles/shf_data")

plot(z0m_data, seriestype=:scatter, ms=1.5, c=:blue, markerstrokecolor=:blue)
png("images/SHEBA_profiles/z0m_data")

plot(z0b_data, seriestype=:scatter, ms=1.5, c=:red, markerstrokecolor=:red)
png("images/SHEBA_profiles/z0b_data")

#=
SUMMARY:
- surface_temp_data
- lhf, shf
- u_star_data
=#