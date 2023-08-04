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

mkpath(joinpath(@__DIR__, "../images"))
mkpath(joinpath(@__DIR__, "../images/phi"))
mkpath(joinpath(@__DIR__, "../data")) # create data folder if not exists
localfile = "data/Stats.cfsite17_CNRM-CM5_amip_2004-2008.10.nc"
data = NCDataset(localfile)

# Extract data
max_z_index = 20

time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["profiles"]["z"])[1:max_z_index] # (200, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"]) # (865, )
u_data = Array(data.group["profiles"]["u_mean"])[1:max_z_index, :] # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"])[1:max_z_index, :] # (200, 865)
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"]) # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"]) # (865, )
L_MO_data = Array(data.group["timeseries"]["obukhov_length_mean"]) # (865, )

Z, T = size(u_data) # dimension variables

# we average all time-dependent quantities temporally
u_star_mean = mean(u_star_data)
u_data = vec(mean(u_data, dims=2))
v_data = vec(mean(v_data, dims=2))
lhf_mean = mean(lhf_data)
shf_mean = mean(shf_data)
L_MO_mean = mean(L_MO_data)
ζ_data = z_data / L_MO_mean

# construct partial u partial z - change in u / change in z from above and below data point averaged
dudz_data = zeros(Z)
Δz = (z_data[Z] - z_data[1])/ (Z - 1)
# first z only uses above data point to calculate gradient
dudz_data[1] = (u_data[2] - u_data[1]) / Δz
# last z only uses below data point to calculate gradient
dudz_data[Z] = (u_data[Z] - u_data[Z - 1]) / Δz
for i in 2:Z-1
    gradient_above = (u_data[i + 1] - u_data[i]) / Δz
    gradient_below = (u_data[i] - u_data[i - 1]) / Δz
    dudz_data[i] = (gradient_above + gradient_below) / 2
end

κ = 0.4
y = (κ * z_data) ./ u_star_mean .* dudz_data

function model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; z, L_MO) = inputs
    L_MO_avg = mean(L_MO)

    overrides = (; a_m, a_h, b_m, b_h)
    _, surf_flux_params = get_surf_flux_params(overrides)

    uft = UF.BusingerType()
    transport = UF.MomentumTransport()

    predicted_phi = zeros(Z)
    for i in 1:Z
        uf = UF.universal_func(uft, L_MO_avg, SFP.uf_params(surf_flux_params))
        ζ = z[i] / L_MO_avg
        predicted_phi[i] = UF.phi(uf, ζ, transport)
    end

    return predicted_phi
end

function G(parameters, inputs)
    return model(parameters, inputs)
end

Γ = 0.05^2 * I * (maximum(y) - minimum(y)) # assume this is the amount of noise in observations y
inputs = (; z = z_data, L_MO = lmo_data)

prior_u1 = constrained_gaussian("a_m", 4.7, 3, 0, Inf)
prior_u2 = constrained_gaussian("a_h", 4.7, 3, 0, Inf)
prior_u3 = constrained_gaussian("a_m", 15.0, 6, 0, Inf)
prior_u4 = constrained_gaussian("a_h", 9.0, 4, 0, Inf)
prior = combine_distributions([prior_u1, prior_u2, prior_u3, prior_u4])

# # Set up the initial ensembles
# N_ensemble = 5
# N_iterations = 5

# rng_seed = 41
# rng = Random.MersenneTwister(rng_seed)
# initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

# # Define EKP and run iterative solver for defined number of iterations
# ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

# for n in 1:N_iterations
#     params_i = get_ϕ_final(prior, ensemble_kalman_process)
#     G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
#     EKP.update_ensemble!(ensemble_kalman_process, G_ens)
# end

# constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
# final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

ENV["GKSwstype"] = "nul"
theta_true = (4.7, 4.7, 15.0, 9.0)
model_truth = model(theta_true, inputs)
ζ_range = z_data / mean(lmo_data)
plot(ζ_range, y, label="y")
plot!(ζ_range, model_truth, label="Model Truth")
xlabel!("ζ")
ylabel!("ϕ(ζ)")
png("images/phi/y_vs_model_truth")
