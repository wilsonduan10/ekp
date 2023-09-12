# Imports
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

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
include("physical_model.jl")

# get data
cfSite = 23
month = 7

data = create_dataframe(cfSite, month)
Z, T = size(data.u)

outputdir = "images/multiple_observables/mo_$(cfSite)_$(month)_0"
for i in 1:length(outputdir)
    if (outputdir[i] == '/')
        mkpath(outputdir[1:i-1])
    end
end
mkpath(outputdir)

# other model parameters
N_observables = 4
parameterTypes = (:b_m, :b_h)
ufpt = UF.BusingerType()
phase_fn = ρTq()
scheme = ValuesOnlyScheme()

flatten(matrix) = vec(reshape(matrix, length(matrix)))
unflatten(vector) = reshape(vector, N_observables, T)

function H(output)
    observables = zeros(N_observables, T)
    for j in 1:T
        sums = zeros(N_observables)
        total = 0
        for i in 1:Z
            if (!isnothing(output[i, j]))
                sums[1] += output[i, j].ustar
                sums[2] += output[i, j].shf
                sums[3] += output[i, j].lhf
                sums[4] += output[i, j].buoy_flux
                total += 1
            end
        end

        for n in 1:N_observables
            observables[n, j] = sums[n] / total
        end
    end
    return flatten(observables)
end

# construct y
observables = ["ustar", "shf", "lhf", "buoy_flux"]
y = zeros(N_observables, T)
y[1, :] = data.u_star
y[2, :] = data.shf
y[3, :] = data.lhf
y[4, :] = data.buoy_flux
y = flatten(y)

function G(parameters)
    Ψ = physical_model(parameters, parameterTypes, data, ufpt, phase_fn, scheme)
    return H(Ψ)
end

variance = 0.05^2 * (maximum(y) - minimum(y)) # assume 5% noise
Γ = variance * I

# Define the prior parameter values which we wish to recover in our pipeline. They are constrained
# to be non-negative due to physical laws, and their mean is given by Businger et al 1971.
prior_u1 = constrained_gaussian("b_m", 15.0, 8, 0, 30)
prior_u2 = constrained_gaussian("b_h", 9.0, 6, 0, 30)
prior = combine_distributions([prior_u1, prior_u2])

N_ensemble = 10
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
    err = get_error(ensemble_kalman_process)[end] #mean((params_true - mean(params_i,dims=2)).^2)
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

# plot model truth
theta_true = (15.0, 9.0)
model_truth = unflatten(G(theta_true))
y = unflatten(y)
for i in 1:N_observables
    plot(y[i, :], c=:green, seriestype=:scatter, ms=5, label="y")
    plot!(model_truth[i, :], c=:red, seriestype=:scatter, ms=5, label="Model truth")
    title!("$(observables[i]) comparison")
    xlabel!("T")
    ylabel!(observables[i])
    png("$(outputdir)/$(observables[i])_model_truth")
end

# plot ensembles
initial = [unflatten(G(constrained_initial_ensemble[:, i])) for i in 1:N_ensemble]
final = [unflatten(G(final_ensemble[:, i])) for i in 1:N_ensemble]
initial_label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
final_label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)

for i in 1:N_observables
    plot(y[i, :], c=:green, seriestype=:scatter, ms=5, label="y")
    for j in 1:N_ensemble
        plot!(initial[j][i, :], c=:red, label="")
        plot!(final[j][i, :], c=:blue, label="")
    end
    xlabel!("T")
    ylabel!(observables[i])
    title!("$(observables[i]) ensembles")
    png("$(outputdir)/$(observables[i])_ensembles")
end