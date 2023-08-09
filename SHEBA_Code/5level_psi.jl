# This file uses ψ as an observable, calculated from data metrics. The model is just the Businger ψ equation
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

using Downloads
using NCDatasets
using CSV, DataFrames

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

include("5level_data.jl")
L_MO_data = Matrix(CSV.read("data/L_MO.csv", DataFrame, header=false, delim='\t'))

# second filter
mask = BitArray(undef, T)
for i in 1:T
    mask[i] = false
end
for i in 1:T
    if (0.0 in u_star_data[:, i] || sum(abs.(L_MO_data[:, i]) .< 0.2) > 0)
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

u_data = filter_matrix2(u_data)
z_data = filter_matrix2(z_data)
u_star_data = filter_matrix2(u_star_data)
L_MO_data = filter_matrix2(L_MO_data)

Z, T = size(z_data)

ζ_data = z_data ./ L_MO_data

# our model is ψ(z0m / L_MO) - ψ(z / L_MO)
function model(parameters, inputs)
    a_m, b_m = parameters
    (; z, L_MO) = inputs
    overrides = (; a_m, b_m)
    _, surf_flux_params = get_surf_flux_params(overrides)

    uft = UF.BusingerType()
    transport = UF.MomentumTransport()

    output = zeros(Z, T)
    for i in 1:Z
        for j in 1:T
            uf = UF.universal_func(uft, L_MO[i, j], SFP.uf_params(surf_flux_params))
            output[i, j] = UF.psi(uf, z0m / L_MO[i, j], transport)

            uf = UF.universal_func(uft, L_MO[i, j], SFP.uf_params(surf_flux_params))
            ζ = z[i, j] / L_MO[i, j]
            output[i, j] -= UF.psi(uf, ζ, transport)
        end
    end
    return vec(reshape(output, Z*T))
end

function G(parameters, inputs)
    return model(parameters, inputs)
end

# construct observable
inputs = (; z = z_data, L_MO = L_MO_data)

z0m = 0.0001
κ = 0.4
y = κ * u_data ./ u_star_data - log.(z_data ./ z0m)
y = vec(reshape(y, Z*T))
Γ = 0.05^2 * I * (maximum(y) - minimum(y))

prior_u1 = constrained_gaussian("a_m", 4.7, 3, 0, Inf)
prior_u2 = constrained_gaussian("b_m", 15.0, 6, 0, Inf)
prior = combine_distributions([prior_u1, prior_u2])

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
ylabel!("ψ(z0m / L_MO) - ψ(z / L_MO)")
mkpath("images/SHEBA_psi")
png("images/SHEBA_psi/y_plot")

theta_true = (4.7, 15.0)
model_truth = model(theta_true, inputs)
plot(reshape(ζ_data, Z*T), model_truth)
plot!(reshape(ζ_data, Z*T), y, seriestype=:scatter, ms=1.5)
xlabel!("ζ")
ylabel!("ψ(z0m / L_MO) - ψ(z / L_MO)")
png("images/SHEBA_psi/y_vs_model_truth")

initial = [model(constrained_initial_ensemble[:, i], inputs) for i in 1:N_ensemble]
final = [model(final_ensemble[:, i], inputs) for i in 1:N_ensemble]
initial_label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
final_label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)

plot(reshape(ζ_data, Z*T), y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter,)
plot!(reshape(ζ_data, Z*T), initial, c = :red, label = initial_label)
plot!(reshape(ζ_data, Z*T), final, c = :blue, label = final_label)
xlabel!("ζ")
ylabel!("ψ(z0m / L_MO) - ψ(z / L_MO)")
png("images/SHEBA_psi/our_plot")

println("INITIAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(constrained_initial_ensemble[1, :])) # [param, ens_no]
println("Mean b_m:", mean(constrained_initial_ensemble[2, :]))
println()

println("FINAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(final_ensemble[1, :])) # [param, ens_no]
println("Mean b_m:", mean(final_ensemble[2, :]))

println("Plots stored in images/SHEBA_psi")
