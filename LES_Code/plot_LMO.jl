# plotting calculated L_MO vs data L_MO
# try changing Fluxes to ValuesOnly
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

include("../helper/setup_parameter_set.jl")

cfsite = 23
month = "07"
localfile = "data/Stats.cfsite$(cfsite)_CNRM-CM5_amip_2004-2008.$(month).nc"
data = NCDataset(localfile)
mkpath(joinpath(@__DIR__, "../images/LES_LMO"))

# We extract the relevant data points for our pipeline.
max_z_index = 5
spin_up = 100

if (max_z_index == 200)
    output_filepath = "images/LES_LMO/LMO_$(cfsite)_$(month)/full"
else
    output_filepath = "images/LES_LMO/LMO_$(cfsite)_$(month)/partial"
end
mkpath(output_filepath)

# profiles
u_data = Array(data.group["profiles"]["u_mean"])[1:max_z_index, spin_up:end]
v_data = Array(data.group["profiles"]["v_mean"])[1:max_z_index, spin_up:end]
qt_data = Array(data.group["profiles"]["qt_mean"])[1:max_z_index, spin_up:end]
θ_li_data = Array(data.group["profiles"]["thetali_mean"])[1:max_z_index, spin_up:end]
temp_data = Array(data.group["profiles"]["temperature_mean"])[1:max_z_index, spin_up:end]

# reference
z_data = Array(data.group["profiles"]["z"])[1:max_z_index]
ρ_data = Array(data.group["reference"]["rho0"])[1:max_z_index]
p_data = Array(data.group["reference"]["p0"])[1:max_z_index]

# timeseries
time_data = Array(data.group["timeseries"]["t"])[spin_up:end]
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"])[spin_up:end]
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"])[spin_up:end]
shf_data = Array(data.group["timeseries"]["shf_surface_mean"])[spin_up:end]
surface_temp_data = Array(data.group["timeseries"]["surface_temperature"])[spin_up:end]
L_MO_data = Array(data.group["timeseries"]["obukhov_length_mean"])[spin_up:end]

Z, T = size(u_data) # extract dimensions for easier indexing

for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
end

unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

function get_LMO(parameters, inputs, info = "Fluxes", ρθq = true)
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, lhf, shf, z0, ustar) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides) # override default Businger params

    # Now, we loop over all the observations and call SF.surface_conditions to estimate u_star
    output = zeros(T)
    for j in 1:T
        L_MO_sum = 0.0
        total = 0
        # Define surface conditions based on moist air density, liquid ice potential temperature, and total specific humidity 
        # given from cfSite. 
        if (ρθq)
            ts_sfc = TD.PhaseEquil_ρθq(thermo_params, ρ_data[1], θ_li_data[1, j], qt_data[1, j]) # use 1 to get surface conditions
        else
            ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_data[1], surface_temp_data[j], qt_data[1, j])
        end
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        # We now loop through all heights at this time step.
        for i in 2:Z # starting at 2 because index 1 is our surface conditions
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            
            if (ρθq)
                ts_in = TD.PhaseEquil_ρθq(thermo_params, ρ_data[i], θ_li_data[i, j], qt_data[i, j])
            else
                ts_in = TD.PhaseEquil_pTq(thermo_params, p_data[i], temp_data[i, j], qt_data[i, j])
            end
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            # We provide a few additional parameters for SF.surface_conditions
            z0m = z0b = z0
            gustiness = FT(1)
            if (info == "Fluxes")
                kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
                # kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness, ustar=ustar[j])
                sc = SF.Fluxes{FT}(; kwargs...)
                # sc = SF.FluxesAndFrictionVelocity{FT}(; kwargs...)
            elseif (info == "ValuesOnly")
                kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
                sc = SF.ValuesOnly{FT}(; kwargs...)
            else
                kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], ustar = ustar[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
                sc = SF.FluxesAndFrictionVelocity{FT}(; kwargs...)
            end

            # Now, we call surface_conditions and store the calculated ustar. We surround it in a try catch
            # to account for unconverged fluxes.
            try
                sf = SF.surface_conditions(surf_flux_params, sc, soltype=RS.VerboseSolution())
                if (sf.L_MO != Inf && sf.L_MO != -Inf)
                    L_MO_sum += sf.L_MO
                    total += 1
                end
            catch e
                println(e)

                z_temp, t_temp = (z_data[i], time_data[j])
                temp_key = (z_temp, t_temp)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end
            
        end

        output[j] = L_MO_sum / max(total, 1)
    end
    return output
end

ENV["GKSwstype"] = "nul"
theta_true = (4.7, 4.7, 15.0, 9.0)
inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data, z0 = 0.0001, ustar = u_star_data)

# Generate plots with ρθq = true
# plot with Fluxes
plot(time_data, L_MO_data, c=:black, label="Data L_MO", legend=:bottomright)
plot!(time_data, get_LMO(theta_true, inputs, "Fluxes", true), label="Model L_MO", seriestype=:scatter, ms=1.5)
title!("Fluxes and ρθq Scheme")
xlabel!("Time")
ylabel!("L_MO")
png("$(output_filepath)/fluxes_1")

# plot with ValuesOnly
plot(time_data, L_MO_data, c=:black, label="Data L_MO", legend=:bottomright)
plot!(time_data, get_LMO(theta_true, inputs, "ValuesOnly", true), label="Model L_MO", seriestype=:scatter, ms=1.5)
title!("ValuesOnly and ρθq Scheme")
xlabel!("Time")
ylabel!("L_MO")
png("$(output_filepath)/values_only_1")

# plot with FluxesAndFrictionVelocity
plot(time_data, L_MO_data, c=:black, label="Data L_MO", legend=:bottomright)
plot!(time_data, get_LMO(theta_true, inputs, "FluxesAndFrictionVelocity", true), label="Model L_MO", seriestype=:scatter, ms=1.5)
title!("FluxesAndFrictionVelocity and ρθq Scheme")
xlabel!("Time")
ylabel!("L_MO")
png("$(output_filepath)/fluxes_and_ustar_1")

# Generate plots with ρθq = false
# plot with Fluxes
plot(time_data, L_MO_data, c=:black, label="Data L_MO", legend=:bottomright)
plot!(time_data, get_LMO(theta_true, inputs, "Fluxes", false), label="Model L_MO", seriestype=:scatter, ms=1.5)
title!("Fluxes and pTq Scheme")
xlabel!("Time")
ylabel!("L_MO")
png("$(output_filepath)/fluxes_2")

# plot with ValuesOnly
plot(time_data, L_MO_data, c=:black, label="Data L_MO", legend=:bottomright)
plot!(time_data, get_LMO(theta_true, inputs, "ValuesOnly", false), label="Model L_MO", seriestype=:scatter, ms=1.5)
title!("ValuesOnly and pTq Scheme")
xlabel!("Time")
ylabel!("L_MO")
png("$(output_filepath)/values_only_2")

# plot with FluxesAndFrictionVelocity
plot(time_data, L_MO_data, c=:black, label="Data L_MO", legend=:bottomright)
plot!(time_data, get_LMO(theta_true, inputs, "FluxesAndFrictionVelocity", false), label="Model L_MO", seriestype=:scatter, ms=1.5)
title!("FluxesAndFrictionVelocity and pTq Scheme")
xlabel!("Time")
ylabel!("L_MO")
png("$(output_filepath)/fluxes_and_ustar_2")
