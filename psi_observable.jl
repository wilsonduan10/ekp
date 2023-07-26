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

ENV["GKSwstype"] = "nul"
mkpath(joinpath(@__DIR__, "images"))
mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
localfile = "data/Stats.cfsite23_CNRM-CM5_amip_2004-2008.01.nc"
data = NCDataset(localfile);

# Construct observables
# try different observables - ustar, L_MO, flux (momentum, heat, buoyancy), or phi
# first try ustar
time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["profiles"]["z"]) # (200, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"]) # (865, ) likely meaned over all z
u_data = Array(data.group["profiles"]["u_mean"]) # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"]) # (200, 865)
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"]) # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"]) # (865, )
lmo_data = Array(data.group["timeseries"]["obukhov_length_mean"]) # (865, )

Z, T = size(u_data) # dimension variables

# use √u^2 + v^2
for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
end

function model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; z, lmo) = inputs
    overrides = (; a_m, a_h, b_m, b_h)
    _, surf_flux_params = get_surf_flux_params(overrides)

    uft = UF.BusingerType()
    transport = UF.MomentumTransport()

    output = zeros(Z, T)
    for j in 1:T
        for i in 1:Z
            uf = UF.universal_func(uft, lmo[j], SFP.uf_params(surf_flux_params))
            ζ = z0m / lmo[j]
            output[i, j] = UF.psi(uf, ζ, transport)

            uf = UF.universal_func(uft, lmo[j], SFP.uf_params(surf_flux_params))
            ζ = z[i] / lmo[j]
            output[i, j] -= UF.psi(uf, ζ, transport)
        end
    end
    return reshape(output, Z * T)
end

function G(parameters, inputs)
    return model(parameters, inputs)
end

# construct observable
z0m = 0.0001
κ = 0.4
y = zeros(Z, T)
for i in 1:Z
    y[i, :] = κ * u_data[i, :] ./ u_star_data .- log(z_data[i] / z0m)
end
y = reshape(y, Z * T)

Γ = 0.005 * I # assume this is the amount of noise in observations y
# η_dist = MvNormal(zeros(Z*T), Γ)
# y = y .+ rand(η_dist)

inputs = (; z = z_data, lmo = lmo_data)

prior_u1 = constrained_gaussian("a_m", 4.7, 3, 0, Inf)
prior_u2 = constrained_gaussian("a_h", 4.7, 3, 0, Inf)
prior_u3 = constrained_gaussian("b_m", 15.0, 8, 0, Inf)
prior_u4 = constrained_gaussian("b_h", 9.0, 6, 0, Inf)
prior = combine_distributions([prior_u1, prior_u2, prior_u3, prior_u4])

# Set up the initial ensembles
N_ensemble = 5
N_iterations = 5

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

# Define EKP and run iterative solver for defined number of iterations
# ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

# for n in 1:N_iterations
#     params_i = get_ϕ_final(prior, ensemble_kalman_process)
#     G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
#     EKP.update_ensemble!(ensemble_kalman_process, G_ens)
# end

# constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
# final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

theta_true = (4.7, 4.7, 15.0, 9.0)
data = model(theta_true, inputs)

y = reshape(y, (Z, T))
data = reshape(data, (Z, T))

function plt(timestep)
    plot(z_data ./ lmo_data[timestep], y[:, timestep], label="y")
    plot!(z_data ./ lmo_data[timestep], data[:, timestep], label="Model Truth")
    xlabel!("ζ")
    ylabel!("-ψ(z/L_MO) + ψ(z0m/L_MO)")
    png("ψ_plot_$(timestep)")
end
plt(1)
plt(200)
plt(400)
plt(600)
plt(800)

# We print the mean parameters of the initial and final ensemble to identify how
# the parameters evolved to fit the dataset. 
# println("INITIAL ENSEMBLE STATISTICS")
# println("Mean a_m:", mean(constrained_initial_ensemble[1, :])) # [param, ens_no]
# println("Mean a_h:", mean(constrained_initial_ensemble[2, :]))
# println("Mean b_m:", mean(constrained_initial_ensemble[3, :]))
# println("Mean b_h:", mean(constrained_initial_ensemble[4, :]))
# println()

# println("FINAL ENSEMBLE STATISTICS")
# println("Mean a_m:", mean(final_ensemble[1, :])) # [param, ens_no]
# println("Mean a_h:", mean(final_ensemble[2, :]))
# println("Mean b_m:", mean(final_ensemble[3, :]))
# println("Mean b_h:", mean(final_ensemble[4, :]))
