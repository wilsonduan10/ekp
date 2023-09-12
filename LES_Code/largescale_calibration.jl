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
    return mean(observable) # average over time
end

# define observable y
y = zeros(K)
Γ = zeros(K, K)
for (k, (site_name, df)) in enumerate(data)
    y[k] = mean(df.u_star)
    variance = 0.1 ^ 2 * (maximum(df.u_star) - minimum(df.u_star))
    Γ[k, k] = variance
end

function G(parameters)
    output = zeros(K)
    for (k, (_, df)) in enumerate(data)
        Ψ = physical_model(parameters, parameterTypes, df, ufpt, phase_fn, scheme)
        output[k] = H(Ψ)
    end
    return output
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
# outputdir = joinpath(@__DIR__, "../images/LES_all")
outputdir = "images/LES_all"
mkpath(outputdir)

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
png("$(outputdir)/y_vs_truth")

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
png("$(outputdir)/ensembles")

outputdir = "images/LES_all/cfSites"
mkpath(outputdir)

for (k, (site_name, df)) in enumerate(data)
    plot(df.time, df.u_star, label="observed", seriestype=:scatter, c=:green, ms=3, markerstroke="green", markershape=:utriangle)
    plot!(df.time, ones(T) .* model_final[k], label="predicted", linewidth = 3, c=:blue)
    plot!(df.time, ones(T) .* mean(df.u_star), label="mean observed", linewidth = 3, c=:orange)
    xlabel!("Time")
    ylabel!("ustar")
    png("$(outputdir)/$(site_name)")
end