# incomplete
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
include("helper/setup_parameter_set.jl")

mkpath(joinpath(@__DIR__, "images"))
mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
localfile = "data/Stats.cfsite17_CNRM-CM5_amip_2004-2008.10.nc"
data = NCDataset(localfile);

# Construct observables
# try different observables - ustar, L_MO, flux (momentum, heat, buoyancy), or phi
# first try ustar
time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["profiles"]["z"]) # (200, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"]) # (865, ) likely meaned over all z
u_data = Array(data.group["profiles"]["u_mean"]) # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"]) # (200, 865)
w_data = Array(data.group["profiles"]["w_mean"]) # (200, 865)
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"]) # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"]) # (865, )
lmo_data = Array(data.group["timeseries"]["obukhov_length_mean"]) # (865, )
# uw_data = Array(data.group["profiles"]["u_sgs_flux_z"]) # (200, 865)
# vw_data = Array(data.group["profiles"]["v_sgs_flux_z"]) # (200, 865)

Z, T = size(u_data) # dimension variables

# use √u^2 + v^2
for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
end

## derive data
# construct u', w'
u_mean = [mean(u_data[i, :]) for i in 1:Z] # (200, )
w_mean = [mean(w_data[i, :]) for i in 1:Z] # (200, )

u_prime_data = zeros(size(u_data))
w_prime_data = zeros(size(w_data))
for i in 1:Z
    u_prime_data[i, :] = u_data[i, :] .- u_mean[i]
    w_prime_data[i, :] = w_data[i, :] .- w_mean[i]
end

# construct u'w'
uw_data = zeros(Z)
for i in 1:Z
    # calculate covariance
    uw_data[i] = 1/(T - 1) * sum(u_prime_data[i, :] .* w_prime_data[i, :])
end

# construct partial u partial z - change in u / change in z from above and below data point averaged
dudz_data = zeros(size(u_data))
Δz = (z_data[Z] - z_data[1])/ (Z - 1)
# first z only uses above data point to calculate gradient
dudz_data[1, :] = (u_data[2, :] .- u_data[1, :]) / Δz
# last z only uses below data point to calculate gradient
dudz_data[Z, :] = (u_data[Z, :] .- u_data[Z - 1, :]) / Δz
for i in 2:Z-1
    gradient_above = (u_data[i + 1, :] .- u_data[i, :]) / Δz
    gradient_below = (u_data[i, :] .- u_data[i - 1, :]) / Δz
    dudz_data[i, :] = (gradient_above .+ gradient_below) / 2
end

# construct observable ϕ(ζ)
# in order make it a function of only ζ, we average out the times for a single dimensional y
dudz_data = reshape(mean(dudz_data, dims=2), Z)
# see equation from spec
κ = 0.4
y = uw_data ./ (κ * z_data) .* dudz_data

function model(parameters, inputs)
    global Z, T
    a_m, a_h = parameters
    (; z, L_MO) = inputs
    L_MO_avg = mean(L_MO)

    overrides = (; a_m, a_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides)

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

Γ = 0.005 * I # assume this is the amount of noise in observations y
inputs = (; z = z_data, L_MO = lmo_data)

prior_u1 = constrained_gaussian("a_m", 4.0, 3, -Inf, Inf);
prior_u2 = constrained_gaussian("a_h", 4.0, 3, -Inf, Inf);
prior = combine_distributions([prior_u1, prior_u2])

# Set up the initial ensembles
N_ensemble = 5;
N_iterations = 5;

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble);

# Define EKP and run iterative solver for defined number of iterations
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

ENV["GKSwstype"] = "nul"
theta_true = (4.7, 4.7)
theta_bad = (100.0, 100.0)
ζ_range = z_data / mean(lmo_data)
plot(
    ζ_range,
    model(theta_true, inputs),
    c = :black,
    label = "Model Truth",
    legend = :bottomright,
    linewidth = 2,
    linestyle = :dash,
)
plot!(
    ζ_range,
    model(theta_bad, inputs),
    c = :blue,
    label = "Model False",
    legend = :bottomright,
    linewidth = 2,
    linestyle = :dot,
)
# plot!(
#     zrange,
#     ones(length(zrange)) .* y[timestep],
#     c = :black,
#     label = "y",
#     legend = :bottomright,
#     linewidth = 2,
#     linestyle = :dot,
# )
# plot!(zrange, ones(length(zrange)) .* u_star_data[timestep], c = :black, label = "Truth u*", legend = :bottomright, linewidth = 2)
# plot!(
#     zrange,
#     [ones(length(zrange)) .* physical_model(initial_ensemble[:, i], inputs)[timestep] for i in 1:N_ensemble],
#     c = :red,
#     label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble), # reshape to convert from vector to matrix
# )
# plot!(
#     zrange,
#     [ones(length(zrange)) .* physical_model(final_ensemble[:, i], inputs)[timestep] for i in 1:N_ensemble],
#     c = :blue,
#     label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble),
# )
xlabel!("ζ")
ylabel!("ϕ(ζ)")
png("images/our_plot")