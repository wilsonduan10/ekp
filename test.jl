using Distributions, Plots
using CLIMAParameters
const CP = CLIMAParameters
FT = Float64

using Downloads
using NCDatasets

import SurfaceFluxes as SF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFP

include("helper/setup_parameter_set.jl")
cfsite = 13
month = "07"
localfile = "data/Stats.cfsite$(cfsite)_CNRM-CM5_amip_2004-2008.$(month).nc"
data = NCDataset(localfile)

z_data = Array(data.group["profiles"]["z"]) # (200, )
p_data = Array(data.group["reference"]["p0"]) # (200, ) Pa
ρ_data = Array(data.group["reference"]["rho0"]) # (200, )
surface_temp_data = Array(data.group["timeseries"]["surface_temperature"]) # K
temp_data = Array(data.group["profiles"]["temperature_mean"]) # (200, 865) # K
qt_data = Array(data.group["profiles"]["qt_mean"]) # (200, 865)

Z, T = size(temp_data) # extract dimensions for easier indexing

ENV["GKSwstype"] = "nul"

thermo_defaults = get_thermodynamic_defaults()
thermo_params = TD.Parameters.ThermodynamicsParameters{FT}(; thermo_defaults...)
R = filter((pair)->pair.first == :gas_constant, thermo_defaults)[1].second
g = filter((pair)->pair.first == :grav, thermo_defaults)[1].second
M = filter((pair)->pair.first == :molmass_dryair, thermo_defaults)[1].second
P_0 = 100000
R = 287.052874
# c_p = 1000

alt_ρ_data = zeros(Z, T)
for j in 1:T
    for i in 1:Z
        ts = TD.PhaseEquil_pTq(thermo_params, p_data[i], temp_data[i, j], qt_data[i, j])
        alt_ρ_data[i, j] = TD.air_density(thermo_params, ts)
    end
end
alt_ρ_data = vec(mean(alt_ρ_data, dims=2))

# calculate virtual temperature
virt_temp_data = zeros(Z, T)
for i in 1:Z
    for j in 1:T
        virt_temp_data[i, j] = TD.virtual_temperature(thermo_params, temp_data[i, j], ρ_data[i])
    end
end

# alt_z_data = zeros(Z)
# for i in 1:Z
#     alt_z_data[i] = mean(virt_temp_data[i, :]) * R / g * log(mean(p_data[1, :]) / mean(p_data[i, :])) * 40
# end

# alt_z_data = zeros(Z)
# alt_z_data[1] = mean(virt_temp_data[1, :]) * R / g * log(P_0 / mean(p_data[1, :]))
# for i in 2:Z
#     alt_z_data[i] = alt_z_data[i-1] + mean(virt_temp_data[i, :]) * R / g * log(mean(p_data[i-1, :]) / mean(p_data[i, :]))
# end

alt_z_data = zeros(Z, T)
alt_z_data[1, :] = virt_temp_data[1, :] * R / g .* log.(P_0 ./ p_data[1, :])
for i in 2:Z
    alt_z_data[i, :] = alt_z_data[i-1, :] .+ virt_temp_data[i, :] * R / g .* log.(p_data[i-1, :] ./ p_data[i, :])
end
alt_z_data = vec(mean(alt_z_data, dims=2))

plot(z_data)
plot!(alt_z_data)
png("test_plot")
