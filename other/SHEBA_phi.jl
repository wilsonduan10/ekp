# same pipeline of Businger calibration but with SHEBA data
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

using Downloads
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

include("helper/setup_parameter_set.jl")
include("helper/graph.jl")

mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists

# get data from data folder
EC_tend_filepath = "data/EC_tend.nc"
ECMWF_filepath = "data/ECMWF.nc"
surf_obs_filepath = "data/surf_obs.nc"

ECMWF_data = NCDataset(ECMWF_filepath)
EC_data = NCDataset(EC_tend_filepath)
surf_obs_data = NCDataset(surf_obs_filepath)

# I index all data from ECMWF and EC_tend starting from 169 in order to align it with the data
# from surf_obs_data, so that all data start on October 29, 1997
# I also reverse the order of the levels such that the first index is the lowest level, and 
# greater levels = greater level instead of before

# timeseries
time_data = Array(surf_obs_data["Jdd"]) # (8112, )
u_star_data = Array(surf_obs_data["ustar"]) # (8112, )
surface_temp_data = Array(surf_obs_data["T_sfc"]) # (8112, )
lhf_data = Array(surf_obs_data["hl"]) # (8112, )
shf_data = Array(surf_obs_data["hs"]) # (8112, )
z0m_data = Array(ECMWF_data["surf-roughness-length"])[169:end] # (8112, )
z0b_data = Array(ECMWF_data["surface-roughness-length-heat"])[169:end] # (8112, )
surface_pressure_data = Array(ECMWF_data["psurf"])[169:end] # (8112, )

# profiles
max_z_index = 1

u_data = Array(EC_data["u"])[end:-1:1, 169:end][1:max_z_index, :]
v_data = Array(EC_data["v"])[end:-1:1, 169:end][1:max_z_index, :]
qv = Array(ECMWF_data["qv"])[end:-1:1, 169:end][1:max_z_index, :]
ql = Array(ECMWF_data["ql"])[end:-1:1, 169:end][1:max_z_index, :]
qi = Array(ECMWF_data["qi"])[end:-1:1, 169:end][1:max_z_index, :]
temp_data = Array(ECMWF_data["T"])[end:-1:1, 169:end][1:max_z_index, :]
p_data = Array(ECMWF_data["p"])[end:-1:1, 169:end][1:max_z_index, :]
qt_data = qv .+ ql .+ qi # (31, 8112)

Z, T = size(u_data) # extract dimensions for easier indexing

# We combine the u and v velocities into a single number to facilitate analysis: u = √u^2 + v^2
for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
end

#=
SUMMARY OF MISSING DATA:
- surface_temp_data
- u_star_data
=#
poor_data = BitArray(undef, T)
for i in 1:T
    if (surface_temp_data[i] == 999.0 || surface_temp_data[i] == 9999.0 || u_star_data[i] == 9999.0 || lhf_data[i] == 9999.0 || shf_data[i] == 9999.0)
        poor_data[i] = 1
    end
end

time_data = time_data[.!poor_data]
u_star_data = u_star_data[.!poor_data]
surface_temp_data = surface_temp_data[.!poor_data] .+ 273.15
surface_pressure_data = surface_pressure_data[.!poor_data]
lhf_data = lhf_data[.!poor_data]
shf_data = shf_data[.!poor_data]
z0m_data = z0m_data[.!poor_data]
z0b_data = z0b_data[.!poor_data]

function filter_matrix(data)
    temp = zeros(Z, length(time_data))
    for i in 1:Z
        temp[i, :] = data[i, :][.!poor_data]
    end
    return temp
end

u_data = filter_matrix(u_data)
qt_data = filter_matrix(qt_data)
temp_data = filter_matrix(temp_data)
p_data = filter_matrix(p_data)

Z, T = size(u_data)

thermo_defaults = get_thermodynamic_defaults()
thermo_params = TD.Parameters.ThermodynamicsParameters{FT}(; thermo_defaults...)
M = filter((pair)->pair.first == :molmass_dryair, thermo_defaults)[1].second
g = filter((pair)->pair.first == :grav, thermo_defaults)[1].second
P_0 = 100000
R = 287.052874
# R = 8.3
c_p = 1000

ρ_data = zeros(Z, T)
for j in 1:T
    for i in 1:Z
        ts = TD.PhaseEquil_pTq(thermo_params, p_data[i, j], temp_data[i, j], qt_data[i, j])
        ρ_data[i, j] = TD.air_density(thermo_params, ts)
    end
end

# calculate virtual temperature
virt_temp_data = zeros(Z, T)
for i in 1:Z
    for j in 1:T
        virt_temp_data[i, j] = TD.virtual_temperature(thermo_params, temp_data[i, j], ρ_data[i, j])
    end
end

# calculate z given other metrics
z_data = zeros(Z, T)
z_data[1, :] = virt_temp_data[1, :] * R / g .* log.(surface_pressure_data ./ p_data[1, :])
for i in 2:Z
    z_data[i, :] = z_data[i-1, :] .+ virt_temp_data[i, :] * R / g .* log.(p_data[i-1, :] ./ p_data[i, :])
end
z_data = vec(mean(z_data, dims=2))

unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

function physical_model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, lhf, shf, z0m_data, z0b_data) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides)

    L_MO_data = zeros(length(time))
    for j in 1:T # 865
        L_MO_sum = 0.0
        total = 0
        ts_sfc = TD.PhaseEquil_pTq(thermo_params, surface_pressure_data[j], surface_temp_data[j], qt_data[1, j]) # use 1 to get surface conditions
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        for i in 1:Z
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i]
            u_in = SVector{2, FT}(u_in, v_in)

            ts_in = TD.PhaseEquil_pTq(thermo_params, p_data[i, j], temp_data[i, j], qt_data[i, j])
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            z0m = z0m_data[j]
            z0b = z0b_data[j]
            gustiness = FT(1)
            # kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
            # sc = SF.Fluxes{FT}(; kwargs...)
            kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
            sc = SF.ValuesOnly{FT}(; kwargs...)

            try
                sf = SF.surface_conditions(surf_flux_params, sc)
                # if (sf.L_MO != -Inf && sf.L_MO != Inf)
                if (sf.L_MO < 500 && sf.L_MO > -500)
                    L_MO_sum += sf.L_MO
                    total += 1
                end
            catch e
                # println(e)

                z_temp, t_temp = (z_data[i], time_data[j])
                temp_key = (z_temp, t_temp)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end
            
        end

        total = max(total, 1) # just in case total is zero, we don't want to divide by 0
        L_MO_data[j] = L_MO_sum / total
    end
    return L_MO_data
end

inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data, z0m_data = z0m_data, z0b_data = z0b_data)
theta_true = (4.7, 4.7, 15.0, 9.0)
L_MO_data = physical_model(theta_true, inputs)
plot(L_MO_data, seriestype=:scatter, ms=1.5)
png("SHEBA_LMO")
