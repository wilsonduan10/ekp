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

outputdir = "images/recover_profile/rp_$(cfSite)_$(month)_0"
for i in 1:length(outputdir)
    if (outputdir[i] == '/')
        mkpath(outputdir[1:i-1])
    end
end
mkpath(outputdir)

# other model parameters
parameterTypes = (:b_m, :b_h)
ufpt = UF.BusingerType()
phase_fn = ρTq()
scheme = ValuesOnlyScheme()

H(output) = vec(reshape(output, length(output)))
H_inverse(output) = reshape(output, 3, Z, T)

function G(parameters)
    Ψ = physical_model_profiles(parameters, parameterTypes, data, ufpt, phase_fn, scheme)
    return H(Ψ)
end

# construct y
y = zeros(3, Z, T)
y[1, :, :] = data.u
y[2, :, :] = data.qt
y[3, :, :] = data.θ
y = H(y)

variance = 0.2^2 * (maximum(y) - minimum(y)) # assume 20% noise
Γ = variance * I

# Define the prior parameter values which we wish to recover in our pipeline. They are constrained
# to be non-negative due to physical laws, and their mean is given by Businger et al 1971.
prior_u1 = constrained_gaussian("b_m", 15.0, 8, 0, 30)
prior_u2 = constrained_gaussian("b_h", 9.0, 6, 0, 30)
prior = combine_distributions([prior_u1, prior_u2])

N_ensemble = 5
N_iterations = 5

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

theta_true = (15, 9)
model_truth = H_inverse(G(theta_true))
u_truth = model_truth[1, :, :]
qt_truth = model_truth[2, :, :]
θ_truth = model_truth[3, :, :]

initial = [H_inverse(G(constrained_initial_ensemble[:, i])) for i in 1:N_ensemble]
final = [H_inverse(G(final_ensemble[:, i])) for i in 1:N_ensemble]
initial_label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
final_label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)

## plot u
function plot_u(z_level)
    # plot model_truth
    plot(data.u[z_level, :], c=:green, seriestype=:scatter, ms=5, label="y", markerstrokewidth=0)
    plot!(u_truth[z_level, :], c=:red, seriestype=:scatter, ms=5, label="Model Truth")
    xlabel!("T")
    ylabel!("Wind speed (u)")
    png("$(outputdir)/u_model_truth")

    # plot ensembles
    plot(data.u[z_level, :], c=:green, seriestype=:scatter, ms=5, label="y", markerstrokewidth=0)
    for i in 1:N_ensemble
        plot!(initial[i][1, z_level, :], c=:red, label="", linewidth=2)
    end
    for i in 1:N_ensemble
        plot!(final[i][1, z_level, :], c=:blue, label="", linewidth=2)
    end
    xlabel!("T")
    ylabel!("Wind speed (u)")
    png("$(outputdir)/u_ensembles")
end
plot_u(1)

## plot qt
function plot_qt(z_level)
    # plot model_truth
    plot(data.qt[z_level, :], c=:green, seriestype=:scatter, ms=5, label="y", markerstrokewidth=0)
    plot!(qt_truth[z_level, :], c=:red, seriestype=:scatter, ms=5, label="Model Truth")
    xlabel!("T")
    ylabel!("Total specific humidity (qt)")
    png("$(outputdir)/qt_model_truth")

    # plot ensembles
    plot(data.qt[z_level, :], c=:green, seriestype=:scatter, ms=5, label="y", markerstrokewidth=0)
    for i in 1:N_ensemble
        plot!(initial[i][2, z_level, :], c=:red, label="", linewidth=2)
    end
    for i in 1:N_ensemble
        plot!(final[i][2, z_level, :], c=:blue, label="", linewidth=2)
    end
    xlabel!("T")
    ylabel!("Total specific humidity (qt)")
    png("$(outputdir)/qt_ensembles")
end
plot_qt(1)

## plot θ
function plot_theta(z_level)
    # plot model_truth
    plot(data.θ[z_level, :], c=:green, seriestype=:scatter, ms=5, label="y", markerstrokewidth=0)
    plot!(θ_truth[z_level, :], c=:red, seriestype=:scatter, ms=5, label="Model Truth")
    xlabel!("T")
    ylabel!("Potential temperature (θ)")
    png("$(outputdir)/θ_model_truth")

    # plot ensembles
    plot(data.θ[z_level, :], c=:green, seriestype=:scatter, ms=5, label="y", markerstrokewidth=0)
    for i in 1:N_ensemble
        plot!(initial[i][3, z_level, :], c=:red, label="", linewidth=2)
    end
    for i in 1:N_ensemble
        plot!(final[i][3, z_level, :], c=:blue, label="", linewidth=2)
    end
    xlabel!("T")
    ylabel!("Potential temperature (θ)")
    png("$(outputdir)/θ_ensembles")
end
plot_theta(1)
