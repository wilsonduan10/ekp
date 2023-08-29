using LinearAlgebra, Random

using Downloads
using NCDatasets
FT = Float64

include("../helper/setup_parameter_set.jl")

Base.@kwdef struct Dataset{FT}
    u::Matrix{FT}
    qt::Matrix{FT}
    temperature::Matrix{FT}
    z::Vector{FT}
    ρ::Vector{FT}
    p::Vector{FT}
    time::Vector{FT}
    u_star::Vector{FT}
    T_sfc::Vector{FT}
    qt_sfc::Vector{FT}
    ρ_sfc::FT
    shf::Vector{FT} = Vector{FT}()
    lhf::Vector{FT} = Vector{FT}()
    θ::Matrix{FT} = zeros(0, 0)
    buoy_flux::Vector{FT} = Vector{FT}()
    L_MO::Vector{FT} = Vector{FT}()
    z0::FT = FT(0.0001)
end

abstract type PhaseEquilFn end
struct ρTq <: PhaseEquilFn end
struct pTq <: PhaseEquilFn end

function create_dataframe(cfsite, month, extrapolate_surface = true)
    if (month < 10)
        month = "0" * string(month)
    end
    localfile = "data/Stats.cfsite$(cfsite)_CNRM-CM5_amip_2004-2008.$(month).nc"
    @assert(isfile(localfile))
    data = NCDataset(localfile)

    max_z_index = 5 # since MOST allows data only in the surface layer
    spin_up = 100

    # profiles
    u_data = Array(data.group["profiles"]["u_mean"])[1:max_z_index, spin_up:end]
    v_data = Array(data.group["profiles"]["v_mean"])[1:max_z_index, spin_up:end]
    qt_data = Array(data.group["profiles"]["qt_mean"])[1:max_z_index, spin_up:end]
    θ_li_data = Array(data.group["profiles"]["thetali_mean"])[1:max_z_index, spin_up:end]
    temp_data = Array(data.group["profiles"]["temperature_mean"])[1:max_z_index, spin_up:end]
    θ_data = Array(data.group["profiles"]["theta_mean"])[1:max_z_index, spin_up:end]

    # reference
    z_data = Array(data.group["reference"]["z"])[1:max_z_index]
    ρ_data = Array(data.group["reference"]["rho0"])[1:max_z_index]
    p_data = Array(data.group["reference"]["p0"])[1:max_z_index]

    # timeseries
    time_data = Array(data.group["timeseries"]["t"])[spin_up:end]
    u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"])[spin_up:end]
    lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"])[spin_up:end]
    shf_data = Array(data.group["timeseries"]["shf_surface_mean"])[spin_up:end]
    T_sfc_data = Array(data.group["timeseries"]["surface_temperature"])[spin_up:end]
    buoy_flux_data = Array(data.group["timeseries"]["buoyancy_flux_surface_mean"])[spin_up:end]
    LMO_data = Array(data.group["timeseries"]["obukhov_length_mean"])[spin_up:end]

    Z, T = size(u_data) # extract dimensions for easier indexing

    # We combine the u and v velocities into a single number to facilitate analysis: u = √u^2 + v^2
    for i in 1:Z
        u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
    end

    # get surface state
    if (extrapolate_surface)
        sfc_input = (; ρ_data, p_data, surface_temp_data = T_sfc_data, temp_data, qt_data)
        ρ_sfc_data, qt_sfc_data = extrapolate_sfc_state(sfc_input, ρTq())
    else
        ρ_sfc_data = ρ_data[1]
        qt_sfc_data = qt_data[1, :]
    end

    # create dataframe
    filtered_data = Dataset{FT}(u=u_data, qt=qt_data, temperature=temp_data, z=z_data, ρ=ρ_data, p=p_data, 
                                time=time_data, u_star=u_star_data, T_sfc=T_sfc_data, qt_sfc=qt_sfc_data, 
                                ρ_sfc = ρ_sfc_data, shf=shf_data, lhf=lhf_data, θ=θ_data, buoy_flux=buoy_flux_data, L_MO=LMO_data)
    return filtered_data
end
