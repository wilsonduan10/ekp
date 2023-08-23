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

get_ts_sfc(thermo_params, data, t, td_state_fn::ρTq) = 
    TD.PhaseEquil_ρTq(thermo_params, data.ρ_sfc, data.T_sfc[t], data.qt_sfc[t])
get_ts_sfc(thermo_params, data, t, td_state_fn::pTq) =
    TD.PhaseEquil_pTq(thermo_params, data.p_sfc, data.T_sfc[t], data.qt_sfc[t])

get_ts_in(thermo_params, data, z, t, td_state_fn::ρTq) =
    TD.PhaseEquil_ρTq(thermo_params, data.ρ[z], data.temperature[z, t], data.qt[z, t])

get_ts_in(thermo_params, data, z, t, td_state_fn::pTq) =
    TD.PhaseEquil_pTq(thermo_params, data.p[z], data.temperature[z, t], data.qt[z, t])

function physical_model(
    parameters::NamedTuple, 
    data, 
    td_state_fn::PhaseEquilFn, 
    asc::SF.AbstractSurfaceConditions{FT} = SF.ValuesOnly{FT}
)
    thermo_params, surf_flux_params = get_surf_flux_params(parameters) # override default Businger params

    Z, T = size(u)
    output = zeros(T)
    for j in 1:T
        sum = 0.0
        total = 0

        # Establish surface conditions
        ts_sfc = get_ts_sfc(thermo_params, data, j, td_state_fn)
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        # We now loop through all heights at this time step.
        for i in 1:Z
            u_in = data.u[i, j]
            v_in = FT(0)
            z_in = data.z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            
            ts_in = get_ts_in(thermo_params, data, i, j, td_state_fn)
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            # We provide a few additional parameters for SF.surface_conditions
            z0m = z0b = z0
            gustiness = FT(1)
            kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
            if (asc == SF.Fluxes{FT})
                kwargs = (; kwargs..., shf = data["shf"][j], lhf = data["lhf"][j])
            elseif (asc == SF.FluxesAndFrictionVelocity{FT})
                kwargs = (; kwargs..., shf = data["shf"][j], lhf = data["lhf"][j], ustar = data["u_star"][j])
            end
            sc = asc(; kwargs...)

            # Now, we call surface_conditions and store the calculated ustar. We surround it in a try catch
            # to account for unconverged fluxes.
            try
                sf = SF.surface_conditions(surf_flux_params, sc, soltype = RS.VerboseSolution())
                sum += sf.ustar
                total += 1
            catch e
                println(e)
            end
        end
        output[j] = sum / total
    end
    return output
end