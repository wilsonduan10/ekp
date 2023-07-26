# plotting calculated L_MO vs data L_MO, but with different PhaseEquil function
# changing Fluxes to ValuesOnly changed a lot
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
cfsite = 10
month = "07"
localfile = "data/Stats.cfsite$(cfsite)_CNRM-CM5_amip_2004-2008.$(month).nc"
data = NCDataset(localfile)

# We extract the relevant data points for our pipeline.
time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["profiles"]["z"]) # (200, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"]) # (865, )
u_data = Array(data.group["profiles"]["u_mean"]) # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"]) # (200, 865)
p_data = Array(data.group["reference"]["p0"]) # (200, )
surface_temp_data = Array(data.group["timeseries"]["surface_temperature"])
temp_data = Array(data.group["profiles"]["temperature_mean"]) # (200, 865)
qt_data = Array(data.group["profiles"]["qt_min"]) # (200, 865)
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"]) # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"]) # (865, )
lmo_data = Array(data.group["timeseries"]["obukhov_length_mean"]) # (865, )

Z, T = size(u_data) # extract dimensions for easier indexing

for i in 1:Z
    for j in 1:T
        u_data[i, j] = sqrt(u_data[i, j] * u_data[i, j] + v_data[i, j] * v_data[i, j])
    end
end


unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

function get_LMO(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, lhf, shf, z0) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides) # override default Businger params

    # Now, we loop over all the observations and call SF.surface_conditions to estimate u_star
    l_mos = zeros(length(time)) # (865, )
    for j in 1:T # 865
        l_mo_sum = 0.0
        total = 0
        # Define surface conditions based on moist air density, liquid ice potential temperature, and total specific humidity 
        # given from cfSite. 
        ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_data[1], surface_temp_data[j], qt_data[1, j])
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)

        for i in 2:Z # starting at 2 because index 1 is our surface conditions
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            
            ts_in = TD.PhaseEquil_ρTq(thermo_params, p_data[i], temp_data[i, j], qt_data[i, j])
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            z0m = z0b = z0
            gustiness = FT(1)
            # kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
            # sc = SF.Fluxes{FT}(; kwargs...)
            kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
            sc = SF.ValuesOnly{FT}(; kwargs...)

            try
                sf = SF.surface_conditions(surf_flux_params, sc, soltype=RS.VerboseSolution())
                if (sf.L_MO != Inf && sf.L_MO != -Inf)
                    l_mo_sum += sf.L_MO
                    total += 1
                end
            catch e
                println(e)

                z_temp, t_temp = (z_data[i], time_data[j])
                temp_key = (z_temp, t_temp)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end
            
        end

        # We average the ustar over all heights and store it.
        total = max(total, 1) # just in case total is zero, we don't want to divide by 0
        l_mos[j] = l_mo_sum / total
    end
    return l_mos
end

ENV["GKSwstype"] = "nul"
theta_true = (4.7, 4.7, 15.0, 9.0)
most_inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data)
z0s = [0.0001]
plot(time_data, lmo_data, c=:black, label="Data L_MO", legend=:bottomright)
for i in 1:length(z0s)
    custom_input = (; most_inputs..., z0 = z0s[i])
    output = get_LMO(theta_true, custom_input)
    plot!(time_data, output, label="z0 = " * string(z0s[i]), linewidth=2)
end
xlabel!("Time")
ylabel!("L_MO")
png("L_MO_plot2")
