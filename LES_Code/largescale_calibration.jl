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

# We include some helper files. The first is to set up the parameters for surface\_conditions, and
# the second is to plot our results.
include("../helper/setup_parameter_set.jl")
include("../helper/graph.jl")

localfile = "data/LES_all.nc"
data = NCDataset(localfile)

K = length(NCDatasets.groupnames(data))
Z, T = (5, 766)
z0 = 0.0001

# Because the model sometimes fails to converge, we store unconverged values in a dictionary
# so we can analyze and uncover the cause of failure.
unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

# We define our physical model. It takes in the parameters a\_m, a\_h, b\_m, b\_h, as well as data
# inputs. It establishes thermodynamic parameters and Businger parameters in order to call the 
# function surface_conditions. We store each time step's u_star and return a list of these u_stars.
function physical_model(parameters)
    b_m, b_h = parameters

    overrides = (; b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides) # override default Businger params

    output = zeros(K)
    for k in 1:K
        groupname = NCDatasets.groupnames(data)[k]
        data_current = data.group[groupname]

        observables = zeros(Z, T)
        for j in 1:T
            # Establish surface conditions
            # ts_sfc = TD.PhaseEquil_ρθq(thermo_params, data_current["rho0"][1], data_current["thetali_mean"][1, j], data_current["qt_mean"][1, j]) # use 1 to get surface conditions
            # ts_sfc = TD.PhaseEquil_pTq(thermo_params, data_current["p0"][1], data_current["surface_temperature"][j], data_current["qt_mean"][1, j])
            ts_sfc = TD.PhaseEquil_pTq(thermo_params, data_current["p0"][1], data_current["temperature_mean"][1, j], data_current["qt_mean"][1, j])
            u_sfc = SVector{2, FT}(FT(0), FT(0))
            # u_sfc = SVector{2, FT}(u[1, j], FT(0))
            # state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)
            state_sfc = SF.SurfaceValues(data_current["z"][1], u_sfc, ts_sfc)

            # We now loop through all heights at this time step.
            for i in 2:Z
                u_in = sqrt(data_current["u_mean"][i, j] ^ 2 + data_current["v_mean"][i, j] ^ 2)
                v_in = FT(0)
                z_in = data_current["z"][i]
                u_in = SVector{2, FT}(u_in, v_in)
                
                # ts_in = TD.PhaseEquil_ρθq(thermo_params, data_current["rho0"][i], data_current["thetali_mean"][i, j], data_current["qt_mean"][i, j])
                ts_in = TD.PhaseEquil_pTq(thermo_params, data_current["p0"][i], data_current["temperature_mean"][i, j], data_current["qt_mean"][i, j])
                state_in = SF.InteriorValues(z_in, u_in, ts_in)

                # We provide a few additional parameters for SF.surface_conditions
                z0m = z0b = z0
                gustiness = FT(1)
                # kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
                # sc = SF.Fluxes{FT}(; kwargs...)
                kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
                sc = SF.ValuesOnly{FT}(; kwargs...)

                # Now, we call surface_conditions and store the calculated ustar. We surround it in a try catch
                # to account for unconverged fluxes.
                try
                    sf = SF.surface_conditions(surf_flux_params, sc, soltype = RS.VerboseSolution())
                    observables[i, j] = sf.ustar
                catch e
                    println(e)
                    z_temp, t_temp = (data_current["z"][i], data_current["t"][j])
                    temp_key = (z_temp, t_temp)
                    haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                    haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                    haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
                end
            end
        end

        output[k] = mean(observables)
    end
    return output
end

# Our function G simply returns the output of the physical model.
function G(parameters)
    return physical_model(parameters)
end

y = zeros(K)
Γ = zeros(K, K)
for k in 1:K
    groupname = NCDatasets.groupnames(data)[k]

    ustars = data.group[groupname]["friction_velocity_mean"]
    y[k] = mean(ustars)
    variance = 0.1 ^ 2 * (maximum(ustars) - minimum(ustars))
    Γ[k, k] = variance
end

# Define the prior parameter values which we wish to recover in our pipeline. They are constrained
# to be non-negative due to physical laws, and their mean is given by Businger et al 1971.
prior_u1 = constrained_gaussian("b_m", 15.0, 8, 0, Inf)
prior_u2 = constrained_gaussian("b_h", 9.0, 6, 0, Inf)
prior = combine_distributions([prior_u1, prior_u2])

N_ensemble = 5
N_iterations = 10

# Define EKP process.
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

# Run EKI for N_iterations
for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m]) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
    err = get_error(ensemble_kalman_process)[end]
    println("Iteration: " * string(n) * ", Error: " * string(err))
end

# We extract the constrained initial and final ensemble for analysis
constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

# Print the unconverged data points to identify a pattern.
if (length(unconverged_data) > 0)
    println("Unconverged data points: ", unconverged_data)
    println("Unconverged z: ", unconverged_z)
    println("Unconverged t: ", unconverged_t)
    println()
end

# We print the mean parameters of the initial and final ensemble to identify how
# the parameters evolved to fit the dataset. 
println("\nINITIAL ENSEMBLE STATISTICS")
println("Mean b_m:", mean(constrained_initial_ensemble[1, :]))
println("Mean b_h:", mean(constrained_initial_ensemble[2, :]))
println()

println("FINAL ENSEMBLE STATISTICS")
println("Mean b_m:", mean(final_ensemble[1, :]))
println("Mean b_h:", mean(final_ensemble[2, :]))

# Generate plots
# output_dir = joinpath(@__DIR__, "../images/LES_all")
output_dir = "images/LES_all"
mkpath(output_dir)

theta_true = (15.0, 9.0)
theta_final = mean(final_ensemble, dims=2)
model_truth = G(theta_true)
model_final = G(theta_final)

# plot y versus the "truth" K predicted u_stars
plot(y, label="y")
plot!(model_truth, label="model_truth")
xlabel!("Location")
ylabel!("Time averaged ustar")
title!("ustar predictions at different cfSites/months")
png("$(output_dir)/y_vs_truth")

# plot initial ensembles vs final ensembles vs y
initial = [G(constrained_initial_ensemble[:, i]) for i in 1:N_ensemble]
final = [G(final_ensemble[:, i]) for i in 1:N_ensemble]
initial_label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
final_label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)

plot(y, c = :green, label = "y", legend = :bottomright, ms = 3, seriestype=:scatter, markerstroke="green", markershape=:utriangle)
plot!(initial, c = :red, label = initial_label)
plot!(final, c = :blue, label = final_label)
xlabel!("Location")
ylabel!("Time averaged ustar")
title!("Ensemble evaluation")
png("$(output_dir)/ensembles")

output_dir = "images/LES_all/cfSites"
mkpath(output_dir)
for k in 1:K
    groupname = NCDatasets.groupnames(data)[k]

    plot(data.group[groupname]["t"], data.group[groupname]["friction_velocity_mean"], label="observed", seriestype=:scatter, c=:green, ms=3, markerstroke="green", markershape=:utriangle)
    plot!(data.group[groupname]["t"], ones(T) .* model_final[k], label="predicted", linewidth = 3, c=:blue)
    plot!(data.group[groupname]["t"], ones(T) .* mean(data.group[groupname]["friction_velocity_mean"]), label="mean observed", linewidth = 3, c=:orange)
    xlabel!("Time")
    ylabel!("ustar")
    png("$(output_dir)/$(groupname)")
end