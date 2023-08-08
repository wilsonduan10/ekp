# Imports
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

using Downloads
using DelimitedFiles
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

# We include some helper files. The first is to set up the parameters for surface\_conditions, and
# the second is to plot our results.
include("../helper/setup_parameter_set.jl")
include("../helper/graph.jl")

include("5level_data.jl")

unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

function physical_model(parameters, inputs, info = "ValuesOnly")
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, z0) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides)

    L_MOs = zeros(Z, T)
    for j in 1:T # 865
        ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_data[j], T_sfc_data[j], q_data[1, j]) # use 1 to get surface conditions
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        for i in 1:Z
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i, j]
            u_in = SVector{2, FT}(u_in, v_in)

            ts_in = TD.PhaseEquil_pTq(thermo_params, p_data[j], T_data[i, j], q_data[i, j])
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            z0m = z0b = z0
            gustiness = FT(1)
            if (info == "ValuesOnly")
                kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
                sc = SF.ValuesOnly{FT}(; kwargs...)
            elseif (info == "Fluxes")
                kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf_data[i, j], lhf = lhf_data[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
                sc = SF.Fluxes{FT}(; kwargs...)
            else
                kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf_data[i, j], lhf = lhf_data[j], ustar = u_star_data[i, j], z0m = z0m, z0b = z0b, gustiness = gustiness)
                sc = SF.FluxesAndFrictionVelocity{FT}(; kwargs...)
            end

            try
                sf = SF.surface_conditions(surf_flux_params, sc)
                L_MOs[i, j] = sf.L_MO
            catch e
                # println(e)

                z_temp, t_temp = (z_data[i, j], time_data[j])
                temp_key = (z_temp, t_temp)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end
            
        end
    end
    return L_MOs
end

inputs = (u = u_data, z = z_data, time = time_data, z0 = 0.0001)
theta_true = (4.7, 4.7, 15.0, 9.0)
model_truth = physical_model(theta_true, inputs, "ValuesOnly")
values_only_truth = physical_model(theta_true, inputs, "ValuesOnly")
fluxes_truth = physical_model(theta_true, inputs, "Fluxes")
fluxes_and_ustar_truth = physical_model(theta_true, inputs, "FluxesAndFrictionVelocity")

for i in 1:Z
    for j in 1:T
        if (values_only_truth[i, j] > 1000)
            values_only_truth[i, j] = 1000
        elseif (values_only_truth[i, j] < -1000)
            values_only_truth[i, j] = -1000
        end
        if (fluxes_truth[i, j] > 1000)
            fluxes_truth[i, j] = 1000
        elseif (fluxes_truth[i, j] < -1000)
            fluxes_truth[i, j] = -1000
        end
        if (fluxes_and_ustar_truth[i, j] > 1000)
            fluxes_and_ustar_truth[i, j] = 1000
        elseif (fluxes_and_ustar_truth[i, j] < -1000)
            fluxes_and_ustar_truth[i, j] = -1000
        end
    end
end

# z1 plot
z = 3
plot(time_data, values_only_truth[z, :], seriestype=:scatter, ms=1.5)
png("test_plot")

plot(time_data, fluxes_truth[z, :], seriestype=:scatter, ms=1.5)
png("test_plot2")

plot(time_data, fluxes_and_ustar_truth[z, :], seriestype=:scatter, ms=1.5)
png("test_plot3")

# write L_MOs to file for external use
# writedlm("data/L_MO.csv", model_truth, '\t')
