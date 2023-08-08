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
include("helper/setup_parameter_set.jl")
include("helper/graph.jl")

mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
data, headers = readdlm("data/hourly_SHEBA.txt", FT, header=true)
headers = vec(headers)
data = transpose(data)

# profiles
z_data = data[5:9, :]
u_star_data = data[55:59, :]
ws_data = data[11:15, :]
wd_data = data[16:20, :]
q_data = data[26:30, :]
shf_data = data[60:64, :]
T_data = data[21:25, :]

# timeseries
time_data = data[1, :]
p_data = data[4, :]
T_sfc_data = data[47, :]
lhf_data = data[65, :] 

Z, T = size(z_data)

# filter out bad data
mask = BitArray(undef, T)
for i in 1:T
    mask[i] = false
end
for i in 1:T
    if (999.0 in z_data[:, i] || 9999.0 in u_star_data[:, i] || 9999.0 in ws_data[:, i] || 
        9999.0 in wd_data[:, i] || 9999.0 in q_data[:, i] || 9999.0 in shf_data[:, i] || 
        9999.0 in T_data[:, i] || 9999.0 in T_sfc_data[i])
        mask[i] = true
    end
end

time_data = time_data[.!mask]
p_data = p_data[.!mask]
T_sfc_data = T_sfc_data[.!mask]
lhf_data = lhf_data[.!mask]

function filter_matrix(param)
    temp = zeros(Z, length(time_data))
    for i in 1:Z
        temp[i, :] = param[i, :][.!mask]
    end
    return temp
end

z_data = filter_matrix(z_data)
q_data = filter_matrix(q_data)
u_star_data = filter_matrix(u_star_data)
ws_data = filter_matrix(ws_data)
wd_data = filter_matrix(wd_data)
shf_data = filter_matrix(shf_data)
T_data = filter_matrix(T_data)

Z, T = size(z_data)

p_data = p_data .* 100 # convert mb to Pa
T_data = T_data .+ 273.15 # convert C to K
T_sfc_data = T_sfc_data .+ 273.15
q_data = q_data .* 0.001 # convert from g/kg to kg/kg

u_data = zeros(Z, T)
# for i in 1:Z
#     u_data[i, :] = ws_data[i, :] .* cos.(deg2rad.(wd_data[i, :]))
# end
v_data = zeros(Z, T)
for i in 1:Z
    u_data[i, :] = ws_data[i, :] .* cos.(deg2rad.(wd_data[i, :]))
    v_data[i, :] = ws_data[i, :] .* sin.(deg2rad.(wd_data[i, :]))
end

for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
end

# construct partial u partial z - change in u / change in z from above and below data point averaged
dudz_data = zeros(Z, T)
# first z only uses above data point to calculate gradient
dudz_data[1, :] = (u_data[2, :] .- u_data[1, :]) ./ max.(1.0, (z_data[2, :] .- z_data[1, :]))
# last z only uses below data point to calculate gradient
dudz_data[Z, :] = (u_data[Z, :] .- u_data[Z - 1, :]) ./ max.(1.0, (z_data[Z, :] .- z_data[Z - 1, :]))
for i in 2:Z-1
    gradient_above = (u_data[i + 1, :] .- u_data[i, :]) ./ max.(1.0, (z_data[i + 1, :] .- z_data[i, :]))
    gradient_below = (u_data[i, :] .- u_data[i - 1, :]) ./ max.(1.0, (z_data[i, :] .- z_data[i - 1, :]))
    dudz_data[i, :] = (gradient_above .+ gradient_below) ./ 2
end

L_MO_data = CSV.read("data/L_MO.csv", DataFrame, header=false, delim='\t')
L_MO_data = Matrix(L_MO_data)
ζ_data = z_data ./ L_MO_data

for i in 1:length(ζ_data)
    if (ζ_data[i] > 100)
        ζ_data[i] = 100
    end
end

for i in 1:Z
    for j in 1:T
        if (u_star_data[i, j] == 0.0)
            u_star_data[i, j] = 0.01
        end
    end
end

function model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; z, L_MO) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
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

prior_u1 = constrained_gaussian("a_m", 4.7, 3, 0, Inf)
prior_u2 = constrained_gaussian("a_h", 4.7, 3, 0, Inf)
prior_u3 = constrained_gaussian("a_m", 15.0, 6, 0, Inf)
prior_u4 = constrained_gaussian("a_h", 9.0, 4, 0, Inf)
prior = combine_distributions([prior_u1, prior_u2, prior_u3, prior_u4])

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
end

constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

plot(reshape(ζ_data, Z*T), y, seriestype=:scatter)
xlabel!("ζ")
ylabel!("ϕ")

mkpath("images/SHEBA_phi")
png("images/SHEBA_phi/test_plot")

println("INITIAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(constrained_initial_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(constrained_initial_ensemble[2, :]))
println("Mean b_m:", mean(constrained_initial_ensemble[3, :]))
println("Mean b_h:", mean(constrained_initial_ensemble[4, :]))
println()

println("FINAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(final_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(final_ensemble[2, :]))
println("Mean b_m:", mean(final_ensemble[3, :]))
println("Mean b_h:", mean(final_ensemble[4, :]))
