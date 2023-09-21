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
include("../helper/setup_parameter_set.jl")
include("../helper/graph.jl")

include("5level_data.jl")

unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

function physical_model(parameters, inputs)
    a_m, a_h = parameters
    (; u, z, time, z0) = inputs

    overrides = (; a_m, a_h)
    toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
    surf_flux_params = create_parameters(toml_dict, ufpt, overrides)
    thermo_params = surf_flux_params.thermo_params

    output = zeros(T)
    for j in 1:T # 865
        ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_data[j], T_sfc_data[j], q_data[1, j]) # use 1 to get surface conditions
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        sum = 0.0
        total = 0

        for i in 2:Z
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
                if (sf.lhf > -100 && sf.lhf < 100)
                    sum += sf.lhf
                    total += 1
                end
                
            catch e
                # println(e)

                z_temp, t_temp = (z_data[i, j], time_data[j])
                temp_key = (z_temp, t_temp)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end
            
        end

        output[j] = sum / max(1, total)

    end
    return output
end

function G(parameters, inputs)
    output = physical_model(parameters, inputs)
    return output
end

inputs = (u = u_data, z = z_data, time = time_data, z0 = 0.0001)
y = lhf_data
variance = 0.05 ^ 2 * (maximum(y) - minimum(y)) # assume 5% variance
Γ = variance * I

prior_u1 = constrained_gaussian("a_m", 4.7, 3, 0, 200)
prior_u2 = constrained_gaussian("a_h", 4.7, 3, 0, 200)
prior = combine_distributions([prior_u1, prior_u2])

N_ensemble = 5
N_iterations = 10

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
    err = get_error(ensemble_kalman_process)[end] #mean((params_true - mean(params_i,dims=2)).^2)
    println("Iteration: " * string(n) * ", Error: " * string(err))
end

constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

if (length(unconverged_data) > 0)
    println("Unconverged data points: ", unconverged_data)
    println("Unconverged z: ", unconverged_z)
    println("Unconverged t: ", unconverged_t)
    println()
end

println("INITIAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(constrained_initial_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(constrained_initial_ensemble[2, :]))
println()

println("FINAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(final_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(final_ensemble[2, :]))

plot_params = (;
    x = time_data,
    y = y,
    observable = shf_data[1, :],
    ax = ("T", "shf"),
    prior = prior,
    model = physical_model,
    inputs = inputs,
    theta_true = (4.7, 4.7),
    ensembles = (constrained_initial_ensemble, final_ensemble),
    N_ensemble = N_ensemble
)

# generate_SHEBA_plots(plot_params, "SHEBA_shf", true)

# plot on same histogram
initial = physical_model(mean(constrained_initial_ensemble, dims=2), inputs)
final = physical_model(mean(final_ensemble, dims=2), inputs)
plot(initial, y, c = :red, legend=:bottomright, label = "Initial Ensemble", ms = 3, seriestype=:scatter, markerstroke="red", alpha = 0.8)
plot!(final, y, c = :blue, label = "Final Ensemble", ms = 3, seriestype=:scatter, markerstroke="blue", alpha = 0.4)

minim = min(minimum(y), minimum(initial), minimum(final))
maxim = max(maximum(y), maximum(initial), maximum(final))
# create dash: y = x
x = minim:0.0001:maxim
plot!(x, x, c=:black, label = "", linestyle=:dash, linewidth=3, seriestype=:path)
xlabel!("Predicted lhf")
ylabel!("Observed lhf")
title!("Ensemble Comparison")
png("test_plot4")