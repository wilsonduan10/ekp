import CLIMAParameters as CP
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import SurfaceFluxes.Parameters as SFP

abstract type PhaseEquilFn end
struct ρTq <: PhaseEquilFn end
struct pTq <: PhaseEquilFn end

function create_uf_parameters(toml_dict, overrides, ::UF.GryanikType)
    FT = CP.float_type(toml_dict)

    aliases = ["Pr_0_Gryanik", "a_m_Gryanik", "a_h_Gryanik", "b_m_Gryanik", "b_h_Gryanik", "ζ_a_Gryanik", "γ_Gryanik"]

    pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    pairs = (; pairs...) # convert to NamedTuple

    pairs = (;
        Pr_0 = pairs.Pr_0_Gryanik,
        a_m = pairs.a_m_Gryanik,
        a_h = pairs.a_h_Gryanik,
        b_m = pairs.b_m_Gryanik,
        b_h = pairs.b_h_Gryanik,
        ζ_a = pairs.ζ_a_Gryanik,
        γ = pairs.γ_Gryanik,
    )
    pairs = override_climaatmos_defaults(pairs, overrides)

    return UF.GryanikParams{FT}(; pairs...)
end

function create_uf_parameters(toml_dict, overrides, ::UF.BusingerType)
    FT = CP.float_type(toml_dict)
    aliases =
        ["Pr_0_Businger", "a_m_Businger", "a_h_Businger", "b_m_Businger", "b_h_Businger", "ζ_a_Businger", "γ_Businger"]

    pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    pairs = (; pairs...) # convert to NamedTuple

    pairs = (;
        Pr_0 = pairs.Pr_0_Businger,
        a_m = pairs.a_m_Businger,
        a_h = pairs.a_h_Businger,
        b_m = pairs.b_m_Businger,
        b_h = pairs.b_h_Businger,
        ζ_a = pairs.ζ_a_Businger,
        γ = pairs.γ_Businger,
    )
    pairs = override_climaatmos_defaults(pairs, overrides)

    return UF.BusingerParams{FT}(; pairs...)
end

function create_uf_parameters(toml_dict, overrides, ::UF.GrachevType)
    FT = CP.float_type(toml_dict)
    aliases = [
        "Pr_0_Grachev",
        "a_m_Grachev",
        "a_h_Grachev",
        "b_m_Grachev",
        "b_h_Grachev",
        "c_h_Grachev",
        "ζ_a_Grachev",
        "γ_Grachev",
    ]

    pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    pairs = (; pairs...) # convert to NamedTuple

    pairs = (;
        Pr_0 = pairs.Pr_0_Grachev,
        a_m = pairs.a_m_Grachev,
        a_h = pairs.a_h_Grachev,
        b_m = pairs.b_m_Grachev,
        b_h = pairs.b_h_Grachev,
        c_h = pairs.c_h_Grachev,
        ζ_a = pairs.ζ_a_Grachev,
        γ = pairs.γ_Grachev,
    )
    pairs = override_climaatmos_defaults(pairs, overrides)

    return UF.GrachevParams{FT}(; pairs...)
end

function create_parameters(toml_dict, ufpt, overrides = (;))
    FT = CP.float_type(toml_dict)

    ufp = create_uf_parameters(toml_dict, overrides, ufpt)
    AUFP = typeof(ufp)

    aliases = string.(fieldnames(TD.Parameters.ThermodynamicsParameters))
    pairs = CP.get_parameter_values!(toml_dict, aliases, "Thermodynamics")
    thermo_params = TD.Parameters.ThermodynamicsParameters{FT}(; pairs...)
    TP = typeof(thermo_params)

    aliases = ["von_karman_const"]
    pairs = CP.get_parameter_values!(toml_dict, aliases, "SurfaceFluxesParameters")
    return SFP.SurfaceFluxesParameters{FT, AUFP, TP}(; pairs..., ufp, thermo_params)
end

function override_climaatmos_defaults(defaults::NamedTuple, overrides::NamedTuple)
    intersect_keys = intersect(keys(defaults), keys(overrides))
    intersect_vals = getproperty.(Ref(overrides), intersect_keys)
    intersect_overrides = (; zip(intersect_keys, intersect_vals)...)
    return merge(defaults, intersect_overrides)
end

function get_thermodynamic_defaults()
    toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
    aliases = string.(fieldnames(TD.Parameters.ThermodynamicsParameters))
    pairs = CP.get_parameter_values!(toml_dict, aliases, "Thermodynamics")
    return pairs
end

"""
    extrapolate_ρ_to_sfc(thermo_params, ts_int, T_sfc)

Uses the ideal gas law and hydrostatic balance to extrapolate for surface density.
"""
function extrapolate_ρ_to_sfc(thermo_params, ts_in, T_sfc)
    T_int = TD.air_temperature(thermo_params, ts_in)
    Rm_int = TD.gas_constant_air(thermo_params, ts_in)
    ρ_air = TD.air_density(thermo_params, ts_in)
    ρ_air * (T_sfc / T_int)^(TD.cv_m(thermo_params, ts_in) / Rm_int)
end

function extrapolate_sfc_state(inputs, td_state_fn::PhaseEquilFn)
    (; ρ_data, p_data, surface_temp_data, temp_data, qt_data) = inputs

    toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
    surf_flux_params = create_parameters(toml_dict, UF.BusingerType())
    thermo_params = surf_flux_params.thermo_params

    Z, T = size(qt_data)
    ρ_output = 0.0
    qt_output = zeros(T)

    for j in 1:T
        for i in 1:Z
            if (typeof(td_state_fn) == ρTq)
                ts_in = TD.PhaseEquil_ρTq(thermo_params, ρ_data[i], temp_data[i, j], qt_data[i, j])
            elseif (typeof(td_state_fn) == pTq)
                ts_in = TD.PhaseEquil_pTq(thermo_params, p_data[i], temp_data[i, j], qt_data[i, j])
            end
            ρ_sfc = extrapolate_ρ_to_sfc(thermo_params, ts_in, surface_temp_data[j])
            q_sfc = TD.q_vap_saturation(thermo_params, surface_temp_data[j], ρ_sfc, TD.PhaseEquil)
            
            ρ_output += ρ_sfc
            qt_output[j] += q_sfc
        end
        qt_output[j] /= Z
    end
    ρ_output /= Z * T
    return ρ_output, qt_output
end
