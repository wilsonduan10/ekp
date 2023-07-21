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

import RootSolvers
const RS = RootSolvers

import SurfaceFluxes as SF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFP
using StaticArrays: SVector
include("helper/setup_parameter_set.jl")
include("helper/graph.jl")

mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
mkpath(joinpath(@__DIR__, "images"))
localfile = "data/Stats.cfsite17_CNRM-CM5_amip_2004-2008.10.nc"
data = NCDataset(localfile)

# Extract data
time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["profiles"]["z"]) # (200, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"]) # (865, )
u_data = Array(data.group["profiles"]["u_mean"]) # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"]) # (200, 865)
ρ_data = Array(data.group["reference"]["rho0_full"]) # (200, )
qt_data = Array(data.group["profiles"]["qt_min"]) # (200, 865)
θ_li_data = Array(data.group["profiles"]["thetali_mean"]) # (200, 865)
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"]) # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"]) # (865, )

Z, T = size(u_data)

# use √u^2 + v^2
for i in 1:Z
    for j in 1:T
        u_data[i, j] = sqrt(u_data[i, j] * u_data[i, j] + v_data[i, j] * v_data[i, j])
    end
end

# store unconverged values, potentially discover pattern
unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

stable = 0
unstable = 0
neutral = 0
function physical_model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, lhf, shf, z0) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides)

    # Now, we loop over all the observations and call SF.surface_conditions to estimate u^*
    u_star = zeros(length(time)) # (865, )
    for j in 1:T # 865
        u_star_sum = 0.0
        total = 0
        ts_sfc = TD.PhaseEquil_ρθq(thermo_params, ρ_data[1], θ_li_data[1, j], qt_data[1, j]) # use 1 to get surface conditions
        for i in 2:Z # 200 - 1, starting at 2 because 1 is surface conditions
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            u_sfc = SVector{2, FT}(FT(0), FT(0))
            
            ts_in = TD.PhaseEquil_ρθq(thermo_params, ρ_data[i], θ_li_data[i, j], qt_data[i, j])

            state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            # We provide a few additional parameters for SF.surface_conditions
            z0m = z0b = z0
            gustiness = FT(1)
            kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
            sc = SF.Fluxes{FT}(; kwargs...)

            # Now, we call surface_conditions and store the calculated ustar:
            try
                sf = SF.surface_conditions(surf_flux_params, sc, soltype=RS.VerboseSolution())
                global stable, unstable, neutral
                sf.L_MO > 0 ? stable += 1 : (sf.L_MO < 0 ? unstable += 1 : neutral += 1)
                u_star_sum += sf.ustar
                total += 1
            catch e
                z_temp, t_temp = (z_data[i], time_data[j])
                temp_key = (z_temp, t_temp)
                println(e)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end
            
        end
        total = max(total, 1) # just in case total is zero, we don't want to divide by 0
        u_star[j] = u_star_sum / total
    end
    return u_star
end

function G(parameters, inputs)
    u_star = physical_model(parameters, inputs) # (865, )
    return u_star
end

inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data, z0 = 0.0001)

Γ = 0.00005 * I
η_dist = MvNormal(zeros(length(u_star_data)), Γ)
y = u_star_data .+ rand(η_dist) # (H ⊙ Ψ ⊙ T^{-1})(θ) + η from Cleary et al 2021

prior_u1 = constrained_gaussian("a_m", 4.0, 4, 0, Inf);
prior_u2 = constrained_gaussian("a_h", 4.0, 4, 0, Inf);
prior_u3 = constrained_gaussian("b_m", 15.0, 8, 0, Inf);
prior_u4 = constrained_gaussian("b_h", 9.0, 6, 0, Inf);
prior = combine_distributions([prior_u1, prior_u2, prior_u3, prior_u4])

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

constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

# plot priors
plot_prior(prior)

# plot good and bad model
theta_true = (4.7, 4.7, 15.0, 9.0)
theta_bad = (100.0, 100.0, 100.0, 100.0)
ax = ("T", "U*")
plot_good_bad_model(time_data, physical_model(theta_true, inputs), physical_model(theta_bad, inputs), ax)

# plot y vs u_star data
plot_noise(time_data, y, u_star_data, ax)

# plot good model and y
plot_y_versus_model(time_data, y, physical_model(theta_true, inputs))

# plot y, good model, and ensembles
initial = [physical_model(constrained_initial_ensemble[:, i], inputs) for i in 1:N_ensemble]
final = [physical_model(final_ensemble[:, i], inputs) for i in 1:N_ensemble]
initial_label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
final_label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
plot_all(time_data, y, physical_model(theta_true, inputs), initial, final, (initial_label, final_label), ax)


# print statistics
println("Stable count: ", stable)
println("Unstable count: ", unstable)
println("Neutral count: ", neutral)

if (length(unconverged_data) > 0)
    println("Unconverged data points: ", unconverged_data)
    println("Unconverged z: ", unconverged_z)
    println("Unconverged t: ", unconverged_t)
    println()
end

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
