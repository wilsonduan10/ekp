# Learrning ψ approximation
# This file uses ψ as an observable, calculated from data metrics
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

import GaussianRandomFields
const GRF = GaussianRandomFields

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
mkpath(joinpath(@__DIR__, "images/function_learning"))
mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
localfile = "data/Stats.cfsite23_CNRM-CM5_amip_2004-2008.01.nc"
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

# use sqrt(u^2 + v^2)
u_data = sqrt.(u_data .* u_data .+ v_data .* v_data)

function G(parameters)
    return parameters
end

# construct observable
z0m = 0.0001
κ = 0.4
y = κ * u_data / u_star_mean - log.(z_data / z0m)
Γ = 0.30^2 * I(length(y)) * (maximum(y) - minimum(y)) # assume 30% noise in observations y

# Define the spatial domain and discretization 
dim = 1
length_scale = 8
pts_per_dim = LinRange(ζ_data[1], ζ_data[Z], Z)
dofs = 30
# smoothness = 1.0
# corr_length = 0.25

grf = GRF.GaussianRandomField(
    GRF.CovarianceFunction(dim, GRF.Exponential(length_scale)),
    # GRF.CovarianceFunction(dim, GRF.Matern(smoothness, corr_length)),
    GRF.KarhunenLoeve(dofs),
    pts_per_dim,
)

distribution = GaussianRandomFieldInterface(grf, GRFJL())
pd = ParameterDistribution(Dict(
    "distribution" => distribution, 
    "name" => "psi", 
    "constraint" => no_constraint()
)) # the fully constrained parameter distribution
prior = pd

N_ensemble = dofs + 2
N_iterations = 10

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_params = construct_initial_ensemble(rng, prior, N_ensemble)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_params, y, Γ, Inversion())

for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m]) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

# plot first 5 initial ensembles
plot()
for i in 1:5
    plot!(ζ_data, constrained_initial_ensemble[:, i])
end
xlabel!("ζ")
ylabel!("ψ(z0m / L_MO) - ψ(z / L_MO)")
png("images/function_learning/initial_ensembles")

# plot first 5 final ensembles
plot()
for i in 1:5
    plot!(ζ_data, final_ensemble[:, i])
end
xlabel!("ζ")
ylabel!("ψ(z0m / L_MO) - ψ(z / L_MO)")
png("images/function_learning/final_ensembles")

# plot mean initial ensemble vs y
plot(ζ_data, vec(mean(constrained_initial_ensemble, dims=2)), label="mean initial ensemble")
plot!(ζ_data, y, seriestype=:scatter, label="y")
xlabel!("ζ")
ylabel!("ψ(z0m / L_MO) - ψ(z / L_MO)")
png("images/function_learning/y_vs_initial_ensemble")

# plot mean final ensemble vs y
plot(ζ_data, vec(mean(final_ensemble, dims=2)), label="mean final ensemble")
plot!(ζ_data, y, seriestype=:scatter, label="y")
xlabel!("ζ")
ylabel!("ψ(z0m / L_MO) - ψ(z / L_MO)")
png("images/function_learning/y_vs_final_ensemble")

println("Stored plots in images/function_learning")
