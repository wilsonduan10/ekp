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
u_data = Array(EC_data["u"])[end:-1:1, 169:end]
v_data = Array(EC_data["v"])[end:-1:1, 169:end]
qv = Array(ECMWF_data["qv"])[end:-1:1, 169:end]
ql = Array(ECMWF_data["ql"])[end:-1:1, 169:end]
qi = Array(ECMWF_data["qi"])[end:-1:1, 169:end]
temp_data = Array(ECMWF_data["T"])[end:-1:1, 169:end]
p_data = Array(ECMWF_data["p"])[end:-1:1, 169:end]
qt_data = qv .+ ql .+ qi # (31, 8112)

Z, T = size(u_data) # extract dimensions for easier indexing

# We combine the u and v velocities into a single number to facilitate analysis: u = √u^2 + v^2
for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
end

#=
SUMMARY OF MISSING DATA:
- surface_temp_data
- lhf, shf - ignore for now since we will try ValuesOnly
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
R = filter((pair)->pair.first == :gas_constant, thermo_defaults)[1].second
M = filter((pair)->pair.first == :molmass_dryair, thermo_defaults)[1].second
g = filter((pair)->pair.first == :grav, thermo_defaults)[1].second
P_0 = 100000
c_p = 1000

# θ_data = zeros(Z, T)
# for i in 1:Z
#     θ_data[i, :] = temp_data[i, :] .* (P_0 ./ p_data[i, :]) .^ (R / c_p)
# end

# TODO: calculate virtual temperature

# T0 = surface_pressure_data .+ 273
z_data = zeros(Z, T)
for i in 1:Z
    z_data[i, :] = temp_data[i, :] * R / g .* log.(surface_pressure_data ./ p_data[i, :])
end

unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

function physical_model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, lhf, shf, z0m_data, z0b_data) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides)

    u_star = zeros(length(time))
    for j in 1:T # 865
        u_star_sum = 0.0
        total = 0
        ts_sfc = TD.PhaseEquil_pTq(thermo_params, surface_pressure_data[j], surface_temp_data[j], qt_data[1, j]) # use 1 to get surface conditions
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        for i in 1:Z
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i, j]
            u_in = SVector{2, FT}(u_in, v_in)

            ts_in = TD.PhaseEquil_pTq(thermo_params, p_data[i, j], temp_data[i, j], qt_data[i, j])
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            z0m = z0m_data[j]
            z0b = z0b_data[j]
            gustiness = FT(1)
            kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
            sc = SF.Fluxes{FT}(; kwargs...)
            # kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
            # sc = SF.ValuesOnly{FT}(; kwargs...)

            try
                sf = SF.surface_conditions(surf_flux_params, sc)
                u_star_sum += sf.ustar
                total += 1
            catch e
                println(e)

                z_temp, t_temp = (z_data[i], time_data[j])
                temp_key = (z_temp, t_temp)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end
            
        end

        total = max(total, 1) # just in case total is zero, we don't want to divide by 0
        u_star[j] = u_star_sum / total
    end
    return u_star
end

function G(parameters, inputs)
    u_star = physical_model(parameters, inputs)
    return u_star
end

inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data, z0m_data = z0m_data, z0b_data = z0b_data)

# The observation data is noisy by default, and we estimate the noise by calculating variance from mean
# May be an overestimate of noise, but that is ok.
variance = 0.0
for u_star in u_star_data
    global variance
    variance += (mean(u_star_data) - u_star) * (mean(u_star_data) - u_star)
end
variance /= T

Γ = variance * I
y = u_star_data

prior_u1 = constrained_gaussian("a_m", 4.7, 3, 0, Inf)
prior_u2 = constrained_gaussian("a_h", 4.7, 3, 0, Inf)
prior_u3 = constrained_gaussian("b_m", 15.0, 8, 0, Inf)
prior_u4 = constrained_gaussian("b_h", 9.0, 6, 0, Inf)
prior = combine_distributions([prior_u1, prior_u2, prior_u3, prior_u4])

N_ensemble = 5
N_iterations = 5

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

plot_params = (;
    x = time_data,
    y = y,
    observable = u_star_data,
    ax = ("T", "U*"),
    prior = prior,
    model = physical_model,
    inputs = inputs,
    theta_true = (4.7, 4.7, 15.0, 9.0),
    theta_bad = (100.0, 100.0, 100.0, 100.0),
    ensembles = (constrained_initial_ensemble, final_ensemble),
    N_ensemble = N_ensemble
)

generate_SHEBA_plots(plot_params, true)

if (length(unconverged_data) > 0)
    println("Unconverged data points: ", unconverged_data)
    println("Unconverged z: ", unconverged_z)
    println("Unconverged t: ", unconverged_t)
    println()
end

println("INITIAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(constrained_initial_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(constrained_initial_ensemble[2, :]))
println("Mean b_m:", mean(constrained_initial_ensemble[3, :]))
println("Mean b_h:", mean(constrained_initial_ensemble[4, :]))
println()

println("FINAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(final_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(final_ensemble[2, :]))
println("Mean b_m:", mean(final_ensemble[3, :]))
println("Mean b_h:", mean(final_ensemble[4, :]))
