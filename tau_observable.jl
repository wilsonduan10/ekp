# comparing y and physical_model(theta_true, inputs), it is clear that this approach is not feasible
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
include("setup_parameter_set.jl")

mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
localfile = "data/Stats.cfsite17_CNRM-CM5_amip_2004-2008.10.nc"
data = NCDataset(localfile)

# Construct observables
time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["profiles"]["z"]) # (200, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"]) # (865, ) likely meaned over all z
u_data = Array(data.group["profiles"]["u_mean"]) # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"]) # (200, 865)
w_data = Array(data.group["profiles"]["w_mean"]) # (200, 865)
ρ_data = Array(data.group["reference"]["rho0_full"]) # (200, )
qt_data = Array(data.group["profiles"]["qt_min"]) # (200, 865)
θ_li_data = Array(data.group["profiles"]["thetali_mean"]) # (200, 865)
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"]) # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"]) # (865, )

Z, T = size(u_data)

# use √u^2 + v^2
for i in 1:size(u_data)[1]
    for j in 1:size(u_data)[2]
        u_data[i, j] = sqrt(u_data[i, j] * u_data[i, j] + v_data[i, j] * v_data[i, j])
    end
end

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

τ_data = zeros(Z)
for i in 1:Z
    τ_data[i] = ρ_data[i] * uw_data[i]
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
    (; u, z, time, lhf, shf) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides)

    fluxes = zeros(Z, T)
    for j in 1:lastindex(time) # 865
        ts_sfc = TD.PhaseEquil_ρθq(thermo_params, ρ_data[1], θ_li_data[1, j], qt_data[1, j]) # use 1 to get surface conditions
        for i in 2:length(z) # 200 - 1, starting at 2 because 1 is surface conditions
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            u_sfc = SVector{2, FT}(FT(0), FT(0))
            
            ts_in = TD.PhaseEquil_ρθq(thermo_params, ρ_data[i], θ_li_data[i, j], qt_data[i, j])

            state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            # We provide a few additional parameters for SF.surface_conditions
            z0m = z0b = FT(0.001)
            gustiness = FT(1)
            kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
            sc = SF.Fluxes{FT}(; kwargs...)

            # Now, we call surface_conditions and store the calculated ustar:
            try
                sf = SF.surface_conditions(surf_flux_params, sc, soltype=RS.VerboseSolution())
                global stable, unstable, neutral
                sf.L_MO > 0 ? stable += 1 : (sf.L_MO < 0 ? unstable += 1 : neutral += 1)
                fluxes[i, j] = sqrt(sf.ρτxz * sf.ρτxz + sf.ρτyz * sf.ρτyz)
            catch e
                z_temp, t_temp = (z_data[i], time_data[j])
                temp_key = (z_temp, t_temp)
                println(e)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end
            
        end
    end
    return mean(fluxes, dims=2)
end

# Here, we define G, which returns observable values given the parameters and inputs
# from the dataset. The observable we elect is the mean of the calculated ustar across
# all z, which is eventually compared to the actual observed ustar.
function G(parameters, inputs)
    fluxes = physical_model(parameters, inputs) # (200, )
    return fluxes
end

inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data)

Γ = mean(τ_data) * mean(τ_data) * I
η_dist = MvNormal(zeros(Z), Γ)
y = τ_data .+ rand(η_dist)

# Assume that users have prior knowledge of approximate truth.
# (e.g. via physical models / subset of obs / physical laws.)
prior_u1 = constrained_gaussian("a_m", 4.0, 3, -Inf, Inf);
prior_u2 = constrained_gaussian("a_h", 4.0, 3, -Inf, Inf);
prior_u3 = constrained_gaussian("b_m", 15.0, 5, 0, Inf);
prior_u4 = constrained_gaussian("b_h", 9.0, 4, 0, Inf);
prior = combine_distributions([prior_u1, prior_u2, prior_u3, prior_u4])

# Set up the initial ensembles
N_ensemble = 5;
N_iterations = 10;

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble);

# Define EKP and run iterative solver for defined number of iterations
# ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

# for n in 1:N_iterations
#     params_i = get_ϕ_final(prior, ensemble_kalman_process)
#     G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
#     EKP.update_ensemble!(ensemble_kalman_process, G_ens)
# end

# final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)


ENV["GKSwstype"] = "nul"
# plot good model vs bad model
theta_true = (4.7, 4.7, 15.0, 9.0)
theta_bad = (100.0, 100.0, 100.0, 100.0)

plot(z_data, y, label="y")
plot!(z_data, physical_model(theta_true, inputs), label="Model Truth")
xlabel!("Z")
ylabel!("τ")
png("our_plot")