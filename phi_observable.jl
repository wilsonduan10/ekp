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

# use √u^2 + v^2
for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
end

## derive data
Z, T = size(u_data) # dimension variables
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
    return thermo_params, surf_flux_params
end

function model(parameters, inputs)
    a_m, a_h = parameters
    (; z, L_MO) = inputs

    overrides = (; a_m, a_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides)

    uft = UF.BusingerType()
    tt = UF.MomentumTransport()
    l_mo = 123
    uf = UF.universal_func(uft, l_mo, SFP.uf_params(surf_flux_params))
end

function G(parameters, inputs)

end