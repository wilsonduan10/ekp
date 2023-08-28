using LinearAlgebra, Random
using Distributions, Plots
FT = Float64

import RootSolvers
const RS = RootSolvers

import SurfaceFluxes as SF
import Thermodynamics as TD
using StaticArrays: SVector

include("../helper/setup_parameter_set.jl")

abstract type PhaseEquilFn end
struct ρTq <: PhaseEquilFn end
struct pTq <: PhaseEquilFn end

struct ValuesOnlyScheme end
struct FluxesScheme end
struct FluxesAndFrictionVelocityScheme end

get_ts_sfc(thermo_params, data, t, ::Union{ρTq, pTq}) = 
    TD.PhaseEquil_ρTq(thermo_params, data.ρ_sfc, data.T_sfc[t], data.qt_sfc[t])

get_ts_in(thermo_params, data, z, t, ::ρTq) =
    TD.PhaseEquil_ρTq(thermo_params, data.ρ[z], data.temperature[z, t], data.qt[z, t])
get_ts_in(thermo_params, data, z, t, ::pTq) =
    TD.PhaseEquil_pTq(thermo_params, data.p[z], data.temperature[z, t], data.qt[z, t])

function physical_model(
    parameters,
    parameterTypes,
    data,
    ufpt::UF.AbstractUniversalFunctionType = UF.BusingerType(),
    td_state_fn::PhaseEquilFn = ρTq(), 
    asc::Union{ValuesOnlyScheme, FluxesScheme, FluxesAndFrictionVelocityScheme} = ValuesOnlyScheme()
)
    @assert(length(parameterTypes) == length(parameters))
    overrides = (; zip(parameterTypes, parameters)...)

    toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
    surf_flux_params = create_parameters(toml_dict, ufpt, overrides)

    Z, T = size(data.u)
    output = Array{Union{SF.SurfaceFluxConditions, Nothing}}(undef, Z, T)
    for j in 1:T
        # Establish surface conditions
        ts_sfc = get_ts_sfc(surf_flux_params.thermo_params, data, j, td_state_fn)
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        # We now loop through all heights at this time step.
        for i in 1:Z
            u_in = data.u[i, j]
            v_in = FT(0)
            z_in = data.z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            
            ts_in = get_ts_in(surf_flux_params.thermo_params, data, i, j, td_state_fn)
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            # We provide a few additional parameters for SF.surface_conditions
            z0m = z0b = data.z0
            gustiness = FT(1)
            kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
            if (typeof(asc) == ValuesOnlyScheme)
                sc = SF.ValuesOnly{FT}(; kwargs...)
            elseif (typeof(asc) == FluxesScheme)
                kwargs = (; kwargs..., shf = data.shf[j], lhf = data.lhf[j])
                sc = SF.Fluxes{FT}(; kwargs...)
            elseif (typeof(asc) == FluxesAndFrictionVelocityScheme)
                kwargs = (; kwargs..., shf = data.shf[j], lhf = data.lhf[j], ustar = data.u_star[j])
                sc = SF.FluxesAndFrictionVelocity{FT}(; kwargs...)
            end

            # Now, we call surface_conditions and store the calculated surface conditions. We surround it in a try catch
            # to account for unconverged fluxes.
            try
                sf = SF.surface_conditions(surf_flux_params, sc, soltype = RS.VerboseSolution())
                output[i, j] = sf
            catch e
                output[i, j] = nothing
                println(e)
            end
        end
    end
    return output
end

function physical_model_profiles(
    parameters,
    parameterTypes,
    data,
    ufpt::UF.AbstractUniversalFunctionType = UF.BusingerType(),
    td_state_fn::PhaseEquilFn = ρTq(), 
    asc::Union{ValuesOnlyScheme, FluxesScheme, FluxesAndFrictionVelocityScheme} = ValuesOnlyScheme(),
    solverScheme = SF.FDScheme()
)
    @assert(length(parameterTypes) == length(parameters))
    overrides = (; zip(parameterTypes, parameters)...)

    toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
    surf_flux_params = create_parameters(toml_dict, ufpt, overrides)

    Z, T = size(data.u)
    P = 3
    output = zeros(P, Z, T)
    for j in 1:T
        # Establish surface conditions
        ts_sfc = get_ts_sfc(surf_flux_params.thermo_params, data, j, td_state_fn)
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        # We now loop through all heights at this time step.
        for i in 1:Z
            u_in = data.u[i, j]
            v_in = FT(0)
            z_in = data.z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            
            ts_in = get_ts_in(surf_flux_params.thermo_params, data, i, j, td_state_fn)
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            # We provide a few additional parameters for SF.surface_conditions
            z0m = z0b = data.z0
            gustiness = FT(1)
            kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
            if (typeof(asc) == ValuesOnlyScheme)
                sc = SF.ValuesOnly{FT}(; kwargs...)
            elseif (typeof(asc) == FluxesScheme)
                kwargs = (; kwargs..., shf = data.shf[j], lhf = data.lhf[j])
                sc = SF.Fluxes{FT}(; kwargs...)
            elseif (typeof(asc) == FluxesAndFrictionVelocityScheme)
                kwargs = (; kwargs..., shf = data.shf[j], lhf = data.lhf[j], ustar = data.u_star[j])
                sc = SF.FluxesAndFrictionVelocity{FT}(; kwargs...)
            end
            
            try
                sf = SF.surface_conditions(surf_flux_params, sc, soltype = RS.VerboseSolution())
                
                # recover u
                X_sfc = 0
                X_star = sf.ustar
                output[1, i, j] = SF.recover_profile(surf_flux_params, sc, sf.L_MO, z_in, X_sfc, X_star, UF.MomentumTransport(), ufpt, solverScheme)
                
                # recover q
                X_sfc = data.qt_sfc[j]
                X_star = SF.compute_physical_scale_coeff(surf_flux_params, sc, sf.L_MO, UF.HeatTransport(), ufpt, SF.FDScheme())
                output2[2, i, j] = SF.recover_profile(surf_flux_params, sc, sf.L_MO, z_in, X_sfc, X_star, UF.HeatTransport(), ufpt, solverScheme)

                # recover theta
                X_sfc = data.T_sfc[j]
                output3[3, i, j] = SF.recover_profile(surf_flux_params, sc, sf.L_MO, z_in, X_sfc, X_star, UF.HeatTransport(), ufpt, solverScheme)
            catch e
                println(e)
            end
        end
    end
    return output
end