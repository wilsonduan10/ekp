# This file uses ψ as an observable, calculated from data metrics. The model is just the Businger ψ equation
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

import SurfaceFluxes as SF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFP
using StaticArrays: SVector

include("../helper/setup_parameter_set.jl")

ENV["GKSwstype"] = "nul"

cfsite = 23
month = "07"
localfile = "data/Stats.cfsite$(cfsite)_CNRM-CM5_amip_2004-2008.$(month).nc"
data = NCDataset(localfile);

# Extract data
max_z_index = 20
spin_up = 100

# profiles
u_data = Array(data.group["profiles"]["u_mean"])[1:max_z_index, spin_up:end] # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"])[1:max_z_index, spin_up:end] # (200, 865)

# reference
z_data = Array(data.group["reference"]["z"])[1:max_z_index] # (200, )

# timeseries
time_data = Array(data.group["timeseries"]["t"])[spin_up:end] # (865, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"])[spin_up:end] # (865, )
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"])[spin_up:end] # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"])[spin_up:end] # (865, )
L_MO_data = Array(data.group["timeseries"]["obukhov_length_mean"])[spin_up:end] # (865, )

Z, T = size(u_data) # dimension variables

# we average all time-dependent quantities temporally
u_star_mean = mean(u_star_data)
u_data = vec(mean(u_data, dims=2))
v_data = vec(mean(v_data, dims=2))
lhf_mean = mean(lhf_data)
shf_mean = mean(shf_data)
L_MO_mean = mean(L_MO_data)
ζ_data = z_data / L_MO_mean

u_data = sqrt.(u_data .* u_data .+ v_data .* v_data)

# our model is ψ(z0m / L_MO) - ψ(z / L_MO)
function model(parameters, inputs)
    b_m, b_h = parameters
    (; z, L_MO_mean) = inputs
    overrides = (; b_m, b_h)
    _, surf_flux_params = get_surf_flux_params(overrides)

    uft = UF.BusingerType()
    transport = UF.MomentumTransport()

    output = zeros(Z)
    for i in 1:Z
        uf = UF.universal_func(uft, L_MO_mean, SFP.uf_params(surf_flux_params))
        output[i] = UF.psi(uf, z0m / L_MO_mean, transport)

        uf = UF.universal_func(uft, L_MO_mean, SFP.uf_params(surf_flux_params))
        ζ = z[i] / L_MO_mean
        output[i] -= UF.psi(uf, ζ, transport)
    end
    return output
end

function G(parameters, inputs)
    return model(parameters, inputs)
end

# construct observable
z0m = 0.0001
κ = 0.4
y = κ * u_data / u_star_mean - log.(z_data / z0m)
Γ = 1.0 * I # assume this is the amount of noise in observations y

inputs = (; z = z_data, L_MO_mean = L_MO_mean)

prior_u1 = constrained_gaussian("b_m", 15.0, 8, 0, Inf)
prior_u2 = constrained_gaussian("b_h", 9.0, 6, 0, Inf)
prior = combine_distributions([prior_u1, prior_u2])

# Set up the initial ensembles
N_ensemble = 5
N_iterations = 5

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

# Define EKP and run iterative solver for defined number of iterations
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(), rng = rng)

for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

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

# plot y versus model truth
theta_true = (15.0, 9.0)
model_truth = model(theta_true, inputs)
plot(ζ_data, y, label="y", c=:green)
plot!(ζ_data, model_truth, label="Model Truth", c=:black)
xlabel!("ζ")
ylabel!("ψ(z0m / L_MO) - ψ(z / L_MO)")
png("images/LES_psi/y_versus_model_truth")

# plot all
plot(ζ_data, y, label="y", c=:green)
plot!(ζ_data, model_truth, label="Model Truth", c=:black)
initial = [model(constrained_initial_ensemble[:, i], inputs) for i in 1:N_ensemble]
final = [model(final_ensemble[:, i], inputs) for i in 1:N_ensemble]
initial_label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
final_label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
plot!(ζ_data, initial, label=initial_label, c=:red)
plot!(ζ_data, final, label=final_label, c=:blue)
xlabel!("ζ")
ylabel!("ψ(z0m / L_MO) - ψ(z / L_MO)")
png("images/LES_psi/ensembles_vs_data")

println("Plots saved in folder images/LES_psi")