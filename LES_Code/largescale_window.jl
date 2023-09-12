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

ENV["GKSwstype"] = "nul"
include("../helper/setup_parameter_set.jl")
include("load_data.jl")

data = Dict()

months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
for cfSite in 17:23
    for month in months
        localfile = "data/Stats.cfsite$(cfSite)_CNRM-CM5_amip_2004-2008.$(month).nc"
        if (isfile(localfile))
            df = create_dataframe(cfSite, month)

            # add to new NCDataset
            groupname = "cfSite_$(cfSite)_month_$(month)"
            data[groupname] = df
        end
    end
end

K = length(data)
Z, T = (5, 766)
time_window = 36
S = ceil(Int, T / time_window)
z0 = 0.0001

# other model parameters
parameterTypes = (:b_m, :b_h)
ufpt = UF.BusingerType()
phase_fn = ρTq()
scheme = ValuesOnlyScheme()
function H(output)
    observable = zeros(T)
    for j in 1:T
        sum = 0.0
        total = 0
        for i in 1:Z
            if (!isnothing(output[i, j]))
                sum += output[i, j].ustar
                total += 1
            end
        end
        observable[j] = sum / total
    end
    return time_average(observable) # average over time
end

function time_average(ustars, calculate_variance = false)
    output = zeros(S)
    variance = zeros(S)
    for i in 1:S
        lower_index = time_window * (i-1) + 1
        upper_index = min(length(ustars), time_window * i)
        output[i] = mean(ustars[lower_index:upper_index])
        if (calculate_variance)
            variance[i] = sum((ustars[lower_index:upper_index] .- output[i]) .^ 2) / (time_window - 1)
        end
    end
    if (calculate_variance)
        return output, variance
    end
    return output
end

# define observable y
y = zeros(K, S)
Γ = zeros(K * S, K * S)
for (k, (site_name, df)) in enumerate(data)
    y[k, :], variances = time_average(df.u_star, true)
    for s in 1:S
        index = (k - 1) * S + s
        Γ[index, index] = variances[s]
    end
end
y = vec(reshape(y, length(y)))

function G(parameters)
    output = zeros(K, S)
    for (k, (_, df)) in enumerate(data)
        Ψ = physical_model(parameters, parameterTypes, df, ufpt, phase_fn, scheme)
        output[k, :] = H(Ψ)
    end
    return vec(reshape(output, length(output)))
end

# Define the prior parameter values which we wish to recover in our pipeline. They are constrained
# to be non-negative due to physical laws, and their mean is given by Businger et al 1971.
prior_u1 = constrained_gaussian("b_m", 15.0, 8, 0, 30)
prior_u2 = constrained_gaussian("b_h", 9.0, 6, 0, 30)
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
output_dir = "images/LES_all_window"
mkpath("images")
mkpath(output_dir)

# plot mean initial ensemble and y histogram
initial_mean = mean(constrained_initial_ensemble, dims=2)
model_initial = G(initial_mean)
plot(y, model_initial, c = :green, label = "", ms = 1.5, seriestype=:scatter)

x = min(minimum(y), minimum(model_initial)):0.0001:max(maximum(y), maximum(model_initial))
plot!(x, x, c=:black, label = "", linestyle=:dash, linewidth=3, seriestype=:path)
xlabel!("Mean Initial Ensemble ustar")
ylabel!("Observed ustar")
title!("y vs Mean Initial Ensemble")
png("$(output_dir)/initial_ensemble")

# plot mean final ensemble and y histogram
final_mean = mean(final_ensemble, dims=2)
model_final = G(final_mean)
plot(y, model_final, c = :green, label = "", ms = 1.5, seriestype=:scatter)

x = min(minimum(y), minimum(model_final)):0.0001:max(maximum(y), maximum(model_final))
plot!(x, x, c=:black, label = "", linestyle=:dash, linewidth=3, seriestype=:path)
xlabel!("Mean FInal Ensemble ustar")
ylabel!("Observed ustar")
title!("y vs Mean Final Ensemble")
png("$(output_dir)/final_ensemble")

# plot all cfsites final ensemble vs y
output_dir = "images/LES_all_window/cfSites"
mkpath(output_dir)

reshaped_model_final = reshape(model_final, (K, S))
for (k, (site_name, df)) in enumerate(data)
    window_prediction = reshaped_model_final[k, :]
    repeated = repeat(window_prediction, time_window)
    repeated_matrix = reshape(repeated, (S, time_window))
    prediction = vec(reshape(transpose(repeated_matrix), length(repeated_matrix)))[1:T]

    plot(df.time, df.u_star, label="observed", seriestype=:scatter, c=:green, ms=3, markerstroke="green", markershape=:utriangle)
    plot!(df.time, prediction, label="predicted", c=:blue, linewidth=3)
    xlabel!("Time")
    ylabel!("ustar")
    png("$(output_dir)/$(site_name)")
end