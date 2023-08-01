# Using LHF as an observable
# Imports
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

mkpath(joinpath(@__DIR__, "../data")) # create data folder if not exists
cfsite = 23
month = "01"
localfile = "data/Stats.cfsite$(cfsite)_CNRM-CM5_amip_2004-2008.$(month).nc"
data = NCDataset(localfile)

# We extract the relevant data points for our pipeline.
max_z_index = 20

time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["profiles"]["z"])[1:max_z_index] # (200, )
u_data = Array(data.group["profiles"]["u_mean"])[1:max_z_index, :] # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"])[1:max_z_index, :] # (200, 865)
ρ_data = Array(data.group["reference"]["rho0"])[1:max_z_index] # (200, )
qt_data = Array(data.group["profiles"]["qt_min"])[1:max_z_index, :] # (200, 865)
p_data = Array(data.group["reference"]["p0"])[1:max_z_index] # (200, )
surface_temp_data = Array(data.group["timeseries"]["surface_temperature"])
temp_data = Array(data.group["profiles"]["temperature_mean"])[1:max_z_index, :] # (200, 865)
θ_li_data = Array(data.group["profiles"]["thetali_mean"])[1:max_z_index, :] # (200, 865)
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"]) # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"]) # (865, )
uw_data = Array(data.group["timeseries"]["uw_surface_mean"]) # (865, )
vw_data = Array(data.group["timeseries"]["vw_surface_mean"]) # (865, )
buoyancy_flux_data = Array(data.group["timeseries"]["buoyancy_flux_surface_mean"]) # (865, )

Z, T = size(u_data) # extract dimensions for easier indexing

# We combine the u and v velocities into a single number to facilitate analysis: u = √u^2 + v^2
for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
    uw_data[i] = sqrt(uw_data[i] * uw_data[i] + vw_data[i] * vw_data[i])
end

# Because the model sometimes fails to converge, we store unconverged values in a dictionary
# so we can analyze and uncover the cause of failure.
unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

function calculate_u(surf_flux_params, sf, z, z0m)
    κ = surf_flux_params.von_karman_const
    output = FT(0)

    output += log(z / z0m)

    uf = UF.universal_func(UF.BusingerType(), sf.L_MO, SFP.uf_params(surf_flux_params))
    output += UF.psi(uf, z0m / L_MO_mean, UF.MomentumTransport())

    uf = UF.universal_func(UF.BusingerType(), sf.L_MO, SFP.uf_params(surf_flux_params))
    output -= UF.psi(uf, z / L_MO_mean, UF.MomentumTransport())

    output *= sf.ustar / κ
end

# We define our physical model. It takes in the parameters a\_m, a\_h, b\_m, b\_h, as well as data
# inputs. It establishes thermodynamic parameters and Businger parameters in order to call the 
# function surface_conditions. We store each time step's u_star and return a list of these u_stars.
function physical_model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, lhf, shf, z0) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides) # override default Businger params

    # Now, we loop over all the observations and call SF.surface_conditions to estimate u_star
    output = zeros(T)
    for j in 1:T # 865
        sum = 0.0
        total = 0 
        # ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_data[1], surface_temp_data[j], qt_data[1, j])
        ts_sfc = TD.PhaseEquil_ρθq(thermo_params, ρ_data[1], θ_li_data[1, j], qt_data[1, j]) # use 1 to get surface conditions
        # ρ_sfc = TD.air_density(thermo_params, ts_sfc)
        # ρ_sfc = ρ_data[1]
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        # We now loop through all heights at this time step.
        for i in 2:Z # starting at 2 because index 1 is our surface conditions
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            
            # ts_in = TD.PhaseEquil_ρTq(thermo_params, p_data[i], temp_data[i, j], qt_data[i, j])
            ts_in = TD.PhaseEquil_ρθq(thermo_params, ρ_data[i], θ_li_data[i, j], qt_data[i, j])
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            # We provide a few additional parameters for SF.surface_conditions
            z0m = z0b = z0
            gustiness = FT(1)
            kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
            sc = SF.Fluxes{FT}(; kwargs...)
            # kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
            # sc = SF.ValuesOnly{FT}(; kwargs...)

            try
                sf = SF.surface_conditions(surf_flux_params, sc)
                u_calc_data = calculate_u(surf_flux_params, sf, z[i], z0m)
                # sum += -sf.ρτxz / ρ_sfc
                # sum += sf.lhf
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

        output[j] = sum / max(total, 1)
    end
    return output
end

# Our function G simply returns the output of the physical model.
function G(parameters, inputs)
    u_star = physical_model(parameters, inputs) # (865, )
    return u_star
end

# Define inputs based on data, to be fed into the physical model.
inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data, z0 = 0.0001)

# The observation data is noisy by default, and we estimate the noise by calculating variance from mean
# May be an overestimate of noise, but that is ok.
variance = 0.0
for x in lhf_data
    global variance
    variance += (mean(lhf_data) - x) * (mean(lhf_data) - x)
end
variance /= T

Γ = variance * I
y = lhf_data

# Define the prior parameter values which we wish to recover in our pipeline. They are constrained
# to be non-negative due to physical laws, and their mean is given by Businger et al 1971.
prior_u1 = constrained_gaussian("a_m", 4.7, 3, 0, Inf)
prior_u2 = constrained_gaussian("a_h", 4.7, 3, 0, Inf)
prior_u3 = constrained_gaussian("b_m", 15.0, 8, 0, Inf)
prior_u4 = constrained_gaussian("b_h", 9.0, 6, 0, Inf)
prior = combine_distributions([prior_u1, prior_u2, prior_u3, prior_u4])

# Hyperparameters: we find it sufficient to define just 5 ensembles and iterations.
N_ensemble = 5
N_iterations = 5

# Define EKP process.
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

# We run ensemble kalman inversrion for N_iterations
# for n in 1:N_iterations
#     params_i = get_ϕ_final(prior, ensemble_kalman_process)
#     G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
#     EKP.update_ensemble!(ensemble_kalman_process, G_ens)
# end

# We extract the constrained initial and final ensemble for analysis.
constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

# In order to plot, we define a set of parameters:
plot_params = (;
    x = time_data,
    y = y,
    ax = ("T", "U*"),
    prior = prior,
    model = physical_model,
    inputs = inputs,
    theta_true = (4.7, 4.7, 15.0, 9.0),
    theta_bad = (100.0, 100.0, 100.0, 100.0),
    ensembles = (constrained_initial_ensemble, final_ensemble),
    N_ensemble = N_ensemble,
    most_inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data),
    z0s = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
)

generate_all_plots(plot_params, "bc", cfsite, month, true)

# Print the unconverged data points to identify a pattern.
if (length(unconverged_data) > 0)
    println("Unconverged data points: ", unconverged_data)
    println("Unconverged z: ", unconverged_z)
    println("Unconverged t: ", unconverged_t)
    println()
end

# We print the mean parameters of the initial and final ensemble to identify how
# the parameters evolved to fit the dataset. 
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

theta_true = (4.7, 4.7, 15.0, 9.0)
model_truth = physical_model(theta_true, inputs)
plot(y, label="y")
plot!(model_truth, label="Model Truth")
title!("lhf comparison")
xlabel!("T")
ylabel!("lhf")
png("lhf_plot")

# theta_bad = (15.0, 15.0, 1.0, 1.0)
# model_bad = physical_model(theta_bad, inputs)
# plot(model_truth, label="Model Truth")
# plot!(model_bad, label="Model Bad")
# title!("model comparison")
# xlabel!("T")
# ylabel!("buoy_flux")
# png("buoy_flux2")