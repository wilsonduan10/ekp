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

unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

function physical_model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, z0) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides)

    L_MOs = zeros(Z, T)
    for j in 1:T # 865
        ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_data[j], T_sfc_data[j], q_data[1, j]) # use 1 to get surface conditions
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        for i in 1:Z
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i, j]
            u_in = SVector{2, FT}(u_in, v_in)

            ts_in = TD.PhaseEquil_pTq(thermo_params, p_data[j], T_data[i, j], q_data[i, j])
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            z0m = z0b = z0
            gustiness = FT(1)
            kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
            sc = SF.ValuesOnly{FT}(; kwargs...)

            try
                sf = SF.surface_conditions(surf_flux_params, sc)
                L_MOs[i, j] = sf.L_MO
            catch e
                # println(e)

                z_temp, t_temp = (z_data[i, j], time_data[j])
                temp_key = (z_temp, t_temp)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end
            
        end
    end
    return L_MOs
end

inputs = (u = u_data, z = z_data, time = time_data, z0 = 0.0001)
theta_true = (4.7, 4.7, 15.0, 9.0)
model_truth = physical_model(theta_true, inputs)

writedlm("data/L_MO.csv", model_truth, '\t')
