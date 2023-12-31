# Imports
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

using Downloads
using DelimitedFiles
using NCDatasets
using CSV, DataFrames

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

include("5level_data.jl")
L_MO_data = Matrix(CSV.read("data/L_MO.csv", DataFrame, header=false, delim='\t'))

# construct partial u partial z - change in u / change in z from above and below data point averaged
dudz_data = zeros(Z, T)
# first z only uses above data point to calculate gradient
dudz_data[1, :] = (u_data[2, :] .- u_data[1, :]) ./ (z_data[2, :] .- z_data[1, :])
# last z only uses below data point to calculate gradient
dudz_data[Z, :] = (u_data[Z, :] .- u_data[Z - 1, :]) ./ (z_data[Z, :] .- z_data[Z - 1, :])
for i in 2:Z-1
    gradient_above = (u_data[i + 1, :] .- u_data[i, :]) ./ (z_data[i + 1, :] .- z_data[i, :])
    gradient_below = (u_data[i, :] .- u_data[i - 1, :]) ./ (z_data[i, :] .- z_data[i - 1, :])
    dudz_data[i, :] = (gradient_above .+ gradient_below) ./ 2
end

# perform second filter
mask = BitArray(undef, T)
for i in 1:T
    mask[i] = false
end
for i in 1:T
    if (0.0 in u_star_data[:, i] || sum(abs.(L_MO_data[:, i]) .< 0.2) > 0 || Inf in dudz_data[:, i]
        || -Inf in dudz_data[:, i])
        mask[i] = true
    end
end

time_data = time_data[.!mask]

function filter_matrix2(data)
    temp = zeros(Z, length(time_data))
    for i in 1:Z
        temp[i, :] = data[i, :][.!mask]
    end
    return temp
end

z_data = filter_matrix2(z_data)
u_star_data = filter_matrix2(u_star_data)
L_MO_data = filter_matrix2(L_MO_data)
dudz_data = filter_matrix2(dudz_data)

Z, T = size(z_data)

ζ_data = z_data ./ L_MO_data

# define model
function model(parameters, inputs)
    (a_m, ) = parameters
    (; z, L_MO) = inputs

    overrides = (; a_m)
    _, surf_flux_params = get_surf_flux_params(overrides)

    uft = UF.BusingerType()
    transport = UF.MomentumTransport()

    predicted_phi = zeros(Z, T)
    for i in 1:Z
        for j in 1:T
            uf = UF.universal_func(uft, L_MO[i, j], SFP.uf_params(surf_flux_params))
            ζ = z[i, j] / L_MO[i, j]
            predicted_phi[i, j] = UF.phi(uf, ζ, transport)
        end
    end

    return vec(reshape(predicted_phi, Z*T))
end

function G(parameters, inputs)
    return model(parameters, inputs)
end

κ = 0.4
y = (κ * z_data) ./ u_star_data .* dudz_data
y = vec(reshape(y, Z*T))
Γ = 0.05^2 * I * (maximum(y) - minimum(y)) # assume this is the amount of noise in observations y

inputs = (; z = z_data, L_MO = L_MO_data)

prior = constrained_gaussian("a_m", 4.7, 3, 0, Inf)

# Set up the initial ensembles
N_ensemble = 5
N_iterations = 5

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

# Define EKP and run iterative solver for defined number of iterations
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
    println(vec(mean(params_i, dims=2)))
end

constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

plot(reshape(ζ_data, Z*T), y, seriestype=:scatter)
xlabel!("ζ")
ylabel!("ϕ")
mkpath("images/SHEBA_phi")
png("images/SHEBA_phi/y_plot")

theta_true = (4.7, )
model_truth = model(theta_true, inputs)
plot(reshape(ζ_data, Z*T), model_truth)
plot!(reshape(ζ_data, Z*T), y, seriestype=:scatter, ms=1.5)
xlabel!("ζ")
ylabel!("ϕ")
png("images/SHEBA_phi/y_vs_model_truth")

initial = [model(constrained_initial_ensemble[:, i], inputs) for i in 1:N_ensemble]
final = [model(final_ensemble[:, i], inputs) for i in 1:N_ensemble]
initial_label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
final_label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)

plot(reshape(ζ_data, Z*T), y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter,)
plot!(reshape(ζ_data, Z*T), initial, c = :red, label = initial_label)
plot!(reshape(ζ_data, Z*T), final, c = :blue, label = final_label)
xlabel!("ζ")
ylabel!("ϕ")
png("images/SHEBA_phi/our_plot")

println("INITIAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(constrained_initial_ensemble[1, :])) # [param, ens_no]
println()

println("FINAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(final_ensemble[1, :])) # [param, ens_no]

println("\nPlots stored in images/SHEBA_phi")

function calculate_error(y, output)
    sum = 0
    for i in 1:length(y)
        sum += (y[i] - output[i]) ^ 2
    end
    return sum
end

truth_error = calculate_error(y, model_truth)
final_error = calculate_error(y, model(vec(mean(final_ensemble, dims=2)), inputs))

println("\nTruth error: ", truth_error)
println("Final ensemble error: ", final_error)
