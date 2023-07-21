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
# try different observables - ustar, L_MO, flux (momentum, heat, buoyancy), or phi
# first try ustar
time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["profiles"]["z"]) # (200, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"]) # (865, ) likely meaned over all z
u_data = Array(data.group["profiles"]["u_mean"]) # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"]) # (200, 865)
ρ_data = Array(data.group["reference"]["rho0_full"]) # (200, )
qt_data = Array(data.group["profiles"]["qt_min"]) # (200, 865)
θ_li_data = Array(data.group["profiles"]["thetali_mean"]) # (200, 865)
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"]) # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"]) # (865, )
# for i in 1:size(u_data)[1]
#     u_data[:, i] = u_data[:, i] * u_star_data[i]
# end

# use √u^2 + v^2
for i in 1:size(u_data)[1]
    for j in 1:size(u_data)[2]
        u_data[i, j] = sqrt(u_data[i, j] * u_data[i, j] + v_data[i, j] * v_data[i, j])
    end
end

# store unconverged values, potentially discover pattern
unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

function get_surf_flux_params(overrides)
    # First, we set up thermodynamic parameters
    ## This line initializes a toml dict, where we will extract parameters from
    toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
    param_set = create_parameters(toml_dict, UF.BusingerType())
    thermo_params = SFP.thermodynamics_params(param_set)

    # initialize κ parameter
    aliases = ["von_karman_const"]
    κ_pairs = CP.get_parameter_values!(toml_dict, aliases, "SurfaceFluxesParameters")
    κ_pairs = (; κ_pairs...)

    # Next, we set up SF parameters
    ## An alias for each constant we need
    aliases = ["Pr_0_Businger", "a_m_Businger", "a_h_Businger", "b_m_Businger", "b_h_Businger", "ζ_a_Businger", "γ_Businger"]
    sf_pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    sf_pairs = (; sf_pairs...) # convert parameter pairs to NamedTuple
    ## change the keys from their alias to more concise keys
    sf_pairs = (;
        Pr_0 = sf_pairs.Pr_0_Businger,
        a_m = sf_pairs.a_m_Businger,
        a_h = sf_pairs.a_h_Businger,
        b_m = sf_pairs.b_m_Businger,
        b_h = sf_pairs.b_h_Businger,
        ζ_a = sf_pairs.ζ_a_Businger,
        γ = sf_pairs.γ_Businger,
    )
    # override default Businger stability function parameters with model parameters
    sf_pairs = override_climaatmos_defaults(sf_pairs, overrides)

    ufp = UF.BusingerParams{FT}(; sf_pairs...) # initialize Businger params

    # Now, we initialize the variable surf_flux_params, which we will eventually pass into 
    # surface_conditions along with mean wind data
    UFP = typeof(ufp)
    TPtype = typeof(thermo_params)
    surf_flux_params = SF.Parameters.SurfaceFluxesParameters{FT, UFP, TPtype}(; κ_pairs..., ufp, thermo_params)
    return thermo_params, surf_flux_params
end

stable = 0
unstable = 0
neutral = 0

function physical_model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, lhf, shf) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides)

    # Now, we loop over all the observations and call SF.surface_conditions to estimate u^*
    u_star = zeros(length(time)) # (865, )
    for j in 1:lastindex(time) # 865
        u_star_sum = 0.0
        total = 0
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

# Here, we define G, which returns observable values given the parameters and inputs
# from the dataset. The observable we elect is the mean of the calculated ustar across
# all z, which is eventually compared to the actual observed ustar.
function G(parameters, inputs)
    u_star = physical_model(parameters, inputs) # (865, )
    return u_star
end

inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data)

Γ = 0.00005 * I
η_dist = MvNormal(zeros(length(u_star_data)), Γ)
y = u_star_data .+ rand(η_dist) # (H ⊙ Ψ ⊙ T^{-1})(θ) + η from Cleary et al 2021

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
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)


ENV["GKSwstype"] = "nul"

# plot good model vs bad model
theta_true = (4.7, 4.7, 15.0, 9.0)
theta_bad = (100.0, 100.0, 100.0, 100.0)
plot(
    time_data,
    physical_model(theta_true, inputs),
    c = :black,
    label = "Model Truth",
    legend = :bottomright,
    ms = 1.5,
    seriestype=:scatter
)
plot!(
    time_data,
    physical_model(theta_bad, inputs),
    c = :red,
    label = "Model Bad",
    legend = :bottomright,
    ms = 1.5,
    seriestype=:scatter
)
xlabel!("T")
ylabel!("U^*")
png("good_bad_model")

# plot y vs u_star data
plot(
    time_data,
    y,
    c = :green,
    label = "y",
    legend = :bottomright,
    ms = 1.5,
    seriestype=:scatter,
)
plot!(time_data, u_star_data, c = :red, label = "Truth u*", ms = 1.5, seriestype=:scatter)
png("y vs ustar")

# plot good model and y
plot(
    time_data,
    physical_model(theta_true, inputs),
    c = :black,
    label = "Model Truth",
    legend = :bottomright,
    ms = 1.5,
    seriestype=:scatter
)
plot!(
    time_data,
    y,
    c = :green,
    label = "y",
    legend = :bottomright,
    ms = 1.5,
    seriestype=:scatter,
)
xlabel!("T")
ylabel!("U^*")
png("good model and y")

# plot y, good model, and ensembles
plot(
    time_data,
    y,
    c = :green,
    label = "y",
    legend = :bottomright,
    ms = 1.5,
    seriestype=:scatter,
)
plot!(
    time_data,
    physical_model(theta_true, inputs),
    c = :black,
    label = "Model Truth",
    legend = :bottomright,
    ms = 1.5,
    seriestype=:scatter
)
plot!(
    time_data,
    [physical_model(initial_ensemble[:, i], inputs) for i in 1:N_ensemble],
    c = :red,
    label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble), # reshape to convert from vector to matrix
)
plot!(
    time_data,
    [physical_model(final_ensemble[:, i], inputs) for i in 1:N_ensemble],
    c = :blue,
    label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble),
)
xlabel!("T")
ylabel!("U^*")
png("our_plot")

println("Mean a_m:", mean(initial_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(initial_ensemble[2, :]))
println("Mean b_m:", mean(initial_ensemble[3, :]))
println("Mean b_h:", mean(initial_ensemble[4, :]))
println()

# Mean values in final ensemble for the two parameters of interest reflect the "truth" within some degree of 
# uncertainty that we can quantify from the elements of `final_ensemble`.
println("Mean a_m:", mean(final_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(final_ensemble[2, :]))
println("Mean b_m:", mean(final_ensemble[3, :]))
println("Mean b_h:", mean(final_ensemble[4, :]))
println()

println("Unconverged data points: ", unconverged_data)
println("Unconverged z: ", unconverged_z)
println("Unconverged t: ", unconverged_t)
println()

println("Stable count: ", stable)
println("Unstable count: ", unstable)
println("Neutral count: ", neutral)
