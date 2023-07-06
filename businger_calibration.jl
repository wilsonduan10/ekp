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
include("setup_parameter_set.jl")

mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
localfile = "data/Stats.cfsite17_CNRM-CM5_amip_2004-2008.10.nc"
data = NCDataset(localfile);

# Construct observables
# try different observables - ustar, L_MO, flux (momentum, heat, buoyancy)
# instead of using u, use √u^2 + v^2
# first try ustar
time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["profiles"]["z"]) # (200, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"]) # (865, ) likely meaned over all z
u_data = Array(data.group["profiles"]["u_mean"]) # (200, 865)
# for i in 1:size(u_data)[1]
#     u_data[:, i] = u_data[:, i] * u_star_data[i]
# end

function get_surf_flux_params(overrides)
    # First, we set up thermodynamic parameters
    ## This line initializes a toml dict, where we will extract parameters from
    toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
    param_set = create_parameters(toml_dict, UF.BusingerType())
    thermo_params = SFP.thermodynamics_params(param_set)

    ## in this idealized case, we assume dry isothermal conditions
    ts_sfc = TD.PhaseEquil_ρθq(thermo_params, FT(1), FT(300), FT(0)) # 1, 300, 0.0
    ts_in = TD.PhaseEquil_ρθq(thermo_params, FT(1.4), FT(500), FT(0.7))

    # initialize κ parameter
    aliases = ["von_karman_const"]
    κ_pairs = CP.get_parameter_values!(toml_dict, aliases, "SurfaceFluxesParameters")
    κ_pairs = (; κ_pairs...)

    # Next, we set up SF parameters
    ## An alias for each constant we need
    aliases = ["Pr_0_Businger", "a_m_Businger", "a_h_Businger", "ζ_a_Businger", "γ_Businger"]
    sf_pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    sf_pairs = (; sf_pairs...) # convert parameter pairs to NamedTuple
    ## change the keys from their alias to more concise keys
    sf_pairs = (;
        Pr_0 = sf_pairs.Pr_0_Businger,
        a_m = sf_pairs.a_m_Businger,
        a_h = sf_pairs.a_h_Businger,
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
    return (surf_flux_params, ts_sfc, ts_in)
end

function physical_model(parameters, inputs)
    a_m, a_h = parameters
    (; u, z, time) = inputs

    overrides = (; a_m, a_h)
    surf_flux_params, ts_sfc, ts_in = get_surf_flux_params(overrides)

    # Now, we loop over all the observations and call SF.surface_conditions to estimate u^*
    u_star = zeros(length(time)) # (865, )
    for j in 1:lastindex(time)
        u_star_sum = 0.0
        for i in 1:size(u_data)[1]
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            u_sfc = SVector{2, FT}(FT(0), FT(0))

            state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            # We provide a few additional parameters for SF.surface_conditions
            z0m = z0b = FT(0.0001)
            gustiness = FT(0)
            kwargs = (; state_in, state_sfc, z0m, z0b, gustiness)
            sc = SF.ValuesOnly{FT}(; kwargs...)

            # Now, we call surface_conditions and store the calculated ustar:
            sf = SF.surface_conditions(surf_flux_params, sc)
            u_star_sum += sf.ustar
        end
        u_star[j] = u_star_sum / length(u)
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

inputs = (u = u_data, z = z_data, time = time_data)

Γ = 0.0005 * I
η_dist = MvNormal(zeros(length(u_star_data)), Γ)
y = u_star_data .+ rand(η_dist) # (H ⊙ Ψ ⊙ T^{-1})(θ) + η from Cleary et al 2021

# Assume that users have prior knowledge of approximate truth.
# (e.g. via physical models / subset of obs / physical laws.)
prior_u1 = constrained_gaussian("a_m", 4.7, 3, -Inf, Inf);
prior_u2 = constrained_gaussian("a_h", 4.7, 3, -Inf, Inf);
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
    ## get_ϕ_final returns the most recently updated constrained parameters, which it used to make the
    ## next model forward and thus the next update 
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    ## calculate the forwarded model values
    G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

# To visualize the success of the inversion, we plot model with 3 different forms of the truth: 
# - The absolute truth of u^* given by the dataset
# - y, the noisy observation we used to calibrate our model parameter κ
# - The output of the physical model given the true κ = 0.4

# We then compare them to the initial ensemble and the final ensemble.
zrange = z_data
timestep = 1
ENV["GKSwstype"] = "nul"
theta_true = (4.7, 4.7)
plot(
    zrange,
    ones(length(zrange)) .* physical_model(theta_true, inputs)[timestep],
    c = :black,
    label = "Model Truth",
    legend = :bottomright,
    linewidth = 2,
    linestyle = :dash,
)
plot!(
    zrange,
    ones(length(zrange)) .* y[timestep],
    c = :black,
    label = "y",
    legend = :bottomright,
    linewidth = 2,
    linestyle = :dot,
)
plot!(zrange, ones(length(zrange)) .* u_star_data[timestep], c = :black, label = "Truth u*", legend = :bottomright, linewidth = 2)
plot!(
    zrange,
    [ones(length(zrange)) .* physical_model(initial_ensemble[:, i], inputs)[timestep] for i in 1:N_ensemble],
    c = :red,
    label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble), # reshape to convert from vector to matrix
)
plot!(
    zrange,
    [ones(length(zrange)) .* physical_model(final_ensemble[:, i], inputs)[timestep] for i in 1:N_ensemble],
    c = :blue,
    label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble),
)
xlabel!("Z")
ylabel!("U^*")
png("our_plot")

# Mean values in final ensemble for the two parameters of interest reflect the "truth" within some degree of 
# uncertainty that we can quantify from the elements of `final_ensemble`.
println("Mean a_m:", mean(final_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(final_ensemble[2, :]))