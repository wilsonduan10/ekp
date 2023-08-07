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
include("helper/setup_parameter_set.jl")
include("helper/graph.jl")

mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
data, headers = readdlm("data/hourly_SHEBA.txt", FT, header=true)
headers = vec(headers)
data = transpose(data)

# profiles
z_data = data[5:9, :]
u_star_data = data[55:59, :]
ws_data = data[11:15, :]
wd_data = data[16:20, :]
q_data = data[26:30, :]
shf_data = data[60:64, :]
T_data = data[21:25, :]

# timeseries
time_data = data[1, :]
p_data = data[4, :]
T_sfc_data = data[47, :]
lhf_data = data[65, :] 

Z, T = size(z_data)

# filter out bad data
mask = BitArray(undef, T)
for i in 1:T
    mask[i] = false
end
for i in 1:T
    if (999.0 in z_data[:, i] || 9999.0 in u_star_data[:, i] || 9999.0 in ws_data[:, i] || 
        9999.0 in wd_data[:, i] || 9999.0 in q_data[:, i] || 9999.0 in shf_data[:, i] || 
        9999.0 in T_data[:, i] || 9999.0 in T_sfc_data[i])
        mask[i] = true
    end
end

time_data = time_data[.!mask]
p_data = p_data[.!mask]
T_sfc_data = T_sfc_data[.!mask]
lhf_data = lhf_data[.!mask]

function filter_matrix(param)
    temp = zeros(Z, length(time_data))
    for i in 1:Z
        temp[i, :] = param[i, :][.!mask]
    end
    return temp
end

z_data = filter_matrix(z_data)
q_data = filter_matrix(q_data)
u_star_data = filter_matrix(u_star_data)
ws_data = filter_matrix(ws_data)
wd_data = filter_matrix(wd_data)
shf_data = filter_matrix(shf_data)
T_data = filter_matrix(T_data)

Z, T = size(z_data)

p_data = p_data .* 100 # convert mb to Pa
T_data = T_data .+ 273.15 # convert C to K
T_sfc_data = T_sfc_data .+ 273.15
q_data = q_data .* 0.001 # convert from g/kg to kg/kg

u_data = zeros(Z, T)
# for i in 1:Z
#     u_data[i, :] = ws_data[i, :] .* cos.(deg2rad.(wd_data[i, :]))
# end
v_data = zeros(Z, T)
for i in 1:Z
    u_data[i, :] = ws_data[i, :] .* cos.(deg2rad.(wd_data[i, :]))
    v_data[i, :] = ws_data[i, :] .* sin.(deg2rad.(wd_data[i, :]))
end

for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
end

# construct partial u partial z - change in u / change in z from above and below data point averaged
dudz_data = zeros(Z, T)
# first z only uses above data point to calculate gradient
dudz_data[1, :] = (u_data[2, :] .- u_data[1, :]) ./ (z_data[2, :] .- z_data[1, :])
# last z only uses below data point to calculate gradient
dudz_data[Z, :] = (u_data[Z, :] .- u_data[Z - 1, :]) ./ (z_data[Z, :] .- z_data[Z - 1, :])
for i in 2:Z-1
    gradient_above = (u_data[i + 1, :] .- u_data[i, :]) ./ (z_data[i + 1, :] .- z_data[i, :])
    gradient_below = (u_data[i, :] .- u_data[i - 1, :]) ./ (z_data[i, :] .- z_data[i - 1, :])
    dudz_data[i, :] = (gradient_above .+ gradient_below) ./ 2
end

# u_star_data = vec(mean(u_star_data, dims=1))
κ = 0.4
y = (κ * z_data) ./ u_star_data .* dudz_data

plot(reshape(z_data, Z*T), reshape(y, Z*T), seriestype=:scatter)
png("test_plot")