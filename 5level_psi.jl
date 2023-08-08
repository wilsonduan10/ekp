# This file uses ψ as an observable, calculated from data metrics. The model is just the Businger ψ equation
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

using Downloads
using NCDatasets
using CSV, DataFrames

using CLIMAParameters
const CP = CLIMAParameters
FT = Float64

import SurfaceFluxes as SF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFP
using StaticArrays: SVector
include("helper/setup_parameter_set.jl")

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

function filter_matrix(data)
    temp = zeros(Z, length(time_data))
    for i in 1:Z
        temp[i, :] = data[i, :][.!mask]
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
v_data = zeros(Z, T)
for i in 1:Z
    u_data[i, :] = ws_data[i, :] .* cos.(deg2rad.(wd_data[i, :]))
    v_data[i, :] = ws_data[i, :] .* sin.(deg2rad.(wd_data[i, :]))
end

for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
end

L_MO_data = CSV.read("data/L_MO.csv", DataFrame, header=false, delim='\t')
L_MO_data = Matrix(L_MO_data)
ζ_data = z_data ./ L_MO_data

for i in 1:length(ζ_data)
    if (ζ_data[i] > 100)
        ζ_data[i] = 100
    end
end

# our model is ψ(z0m / L_MO) - ψ(z / L_MO)
function model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; z, L_MO_mean) = inputs
    overrides = (; a_m, a_h, b_m, b_h)
    _, surf_flux_params = get_surf_flux_params(overrides)

    uft = UF.BusingerType()
    transport = UF.MomentumTransport()

    output = zeros(Z)
    for i in 1:Z
        uf = UF.universal_func(uft, L_MO_mean, SFP.uf_params(surf_flux_params))
        output[i] = UF.psi(uf, z0m / L_MO_mean, transport)

        uf = UF.universal_func(uft, L_MO_mean, SFP.uf_params(surf_flux_params))
        ζ = z[i] / L_MO_mean
        output[i] -= UF.psi(uf, ζ, transport)
    end
    return output
end

# construct observable
z0m = 0.0001
κ = 0.4
y = κ * u_data ./ u_star_data - log.(z_data ./ z0m)

plot(reshape(ζ_data, Z*T), reshape(y, Z*T), seriestype=:scatter)
xlabel!("ζ")
ylabel!("ψ")

mkpath("images/SHEBA_psi")
png("images/SHEBA_psi/test_plot")
