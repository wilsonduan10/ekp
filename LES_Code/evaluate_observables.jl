using LinearAlgebra, Random
using Distributions, Plots
using Downloads
using NCDatasets

using CLIMAParameters
const CP = CLIMAParameters
FT = Float64

import SurfaceFluxes as SF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFP
using StaticArrays: SVector

include("../helper/setup_parameter_set.jl")

mkpath(joinpath(@__DIR__, "../data")) # create data folder if not exists
cfsite = 10
month = "07"
localfile = "data/Stats.cfsite$(cfsite)_CNRM-CM5_amip_2004-2008.$(month).nc"
mkpath(joinpath(@__DIR__, "../images/LES_observables/observables_$(cfsite)_$(month)"))
data = NCDataset(localfile)

# We extract the relevant data points for our pipeline.
max_z_index = 5
spin_up = 100

# profiles
u_data = Array(data.group["profiles"]["u_mean"])[1:max_z_index, spin_up:end]
v_data = Array(data.group["profiles"]["v_mean"])[1:max_z_index, spin_up:end]
qt_data = Array(data.group["profiles"]["qt_mean"])[1:max_z_index, spin_up:end]
θ_li_data = Array(data.group["profiles"]["thetali_mean"])[1:max_z_index, spin_up:end]
temp_data = Array(data.group["profiles"]["temperature_mean"])

# reference
z_data = Array(data.group["profiles"]["z"])[1:max_z_index]
ρ_data = Array(data.group["reference"]["rho0"])[1:max_z_index]
p_data = Array(data.group["reference"]["p0"])

# timeseries
time_data = Array(data.group["timeseries"]["t"])[spin_up:end]
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"])[spin_up:end]
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"])[spin_up:end]
shf_data = Array(data.group["timeseries"]["shf_surface_mean"])[spin_up:end]
surface_temp_data = Array(data.group["timeseries"]["surface_temperature"])[spin_up:end]
uw_data = Array(data.group["timeseries"]["uw_surface_mean"])[spin_up:end]
vw_data = Array(data.group["timeseries"]["vw_surface_mean"])[spin_up:end]
buoyancy_flux_data = Array(data.group["timeseries"]["buoyancy_flux_surface_mean"])[spin_up:end]

Z, T = size(u_data) # extract dimensions for easier indexing

# We combine the u and v velocities into a single number to facilitate analysis: u = √u^2 + v^2
for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
    uw_data[i] = sqrt(uw_data[i] * uw_data[i] + vw_data[i] * vw_data[i])
end

ρ_m = mean(ρ_data[1, :])
τ_data = -ρ_m .* uw_data

function physical_model(parameters, inputs, pTq = false)
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, lhf, shf, z0) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides) # override default Businger params

    u_star_output = zeros(T)
    shf_output = zeros(T)
    lhf_output = zeros(T)
    buoy_output = zeros(T)
    tau_output = zeros(T)
    evaporation_output = zeros(T)
    for j in 1:T # 865
        sums = zeros(6) # for 7 outputs
        total = 0
        if (pTq)
            ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_data[1], surface_temp_data[j], qt_data[1, j])
            ρ_sfc = TD.air_density(thermo_params, ts_sfc)
        else 
            ts_sfc = TD.PhaseEquil_ρθq(thermo_params, ρ_data[1], θ_li_data[1, j], qt_data[1, j]) # use 1 to get surface conditions
            ρ_sfc = ρ_data[1]
        end
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        # We now loop through all heights at this time step.
        for i in 2:Z # starting at 2 because index 1 is our surface conditions
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            
            if (pTq)
                ts_in = TD.PhaseEquil_ρTq(thermo_params, p_data[i], temp_data[i, j], qt_data[i, j])
            else
                ts_in = TD.PhaseEquil_ρθq(thermo_params, ρ_data[i], θ_li_data[i, j], qt_data[i, j])
            end
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            # We provide a few additional parameters for SF.surface_conditions
            z0m = z0b = z0
            gustiness = FT(1)
            # kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
            # sc = SF.Fluxes{FT}(; kwargs...)
            kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
            sc = SF.ValuesOnly{FT}(; kwargs...)

            try
                sf = SF.surface_conditions(surf_flux_params, sc)
                sums[1] += sf.ustar
                sums[2] += sf.shf
                sums[3] += sf.lhf
                sums[4] += sf.buoy_flux
                sums[5] += sf.ρτxz
                sums[6] += sf.evaporation
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
        u_star_output[j] = sums[1] / total
        shf_output[j] = sums[2] / total
        lhf_output[j] = sums[3] / total
        buoy_output[j] = sums[4] / total
        tau_output[j] = sums[5] / total
        evaporation_output[j] = sums[6] / total
    end
    return (u_star_output, shf_output, lhf_output, buoy_output, tau_output, evaporation_output)
end

inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data, z0 = 0.0001)
theta_true = (4.7, 4.7, 15.0, 9.0)
u_star_model, shf_model, lhf_model, buoy_model, tau_model, evaporation_model = physical_model(theta_true, inputs)
u_star_model2, shf_model2, lhf_model2, buoy_model2, tau_model2, evaporation_model2 = physical_model(theta_true, inputs, true)

function generate_plot(y, model_truth, model_truth_pTq, name)
    plot(y, label="y")
    plot!(model_truth, label="Model Truth")
    xlabel!("T")
    ylabel!(name)
    title!("$(name) Comparison (ρθq)")
    png("images/LES_observables/observables_$(cfsite)_$(month)/$(name)_ρθq")

    plot(y, label="y")
    plot!(model_truth_pTq, label="Model Truth")
    xlabel!("T")
    ylabel!(name)
    title!("$(name) Comparison (pTq)")
    png("images/LES_observables/observables_$(cfsite)_$(month)/$(name)_pTq")
end

generate_plot(u_star_data, u_star_model, u_star_model2, "u_star")
generate_plot(shf_data, shf_model, shf_model2, "shf")
generate_plot(lhf_data, lhf_model, lhf_model2, "lhf")
generate_plot(buoyancy_flux_data, buoy_model, buoy_model2, "buoyancy_flux")
generate_plot(τ_data, tau_model, tau_model2, "τ")
generate_plot(vec(mean(qt_data, dims=1)), evaporation_model, evaporation_model2, "evaporation")
println("Plots generated in folder: images/LES_observables/observables_$(cfsite)_$(month)")
