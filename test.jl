using Distributions, Plots
using CLIMAParameters
const CP = CLIMAParameters
FT = Float64

import SurfaceFluxes as SF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFP

include("helper/setup_parameter_set.jl")

z_data = Array(data.group["profiles"]["z"]) # (200, )
p_data = Array(data.group["reference"]["p0"]) # (200, )
ρ_data = Array(data.group["reference"]["rho0"]) # (200, )
surface_temp_data = Array(data.group["timeseries"]["surface_temperature"])
temp_data = Array(data.group["profiles"]["temperature_mean"]) # (200, 865)
qt_data = Array(data.group["profiles"]["qt_min"]) # (200, 865)

Z, T = size(temp_data) # extract dimensions for easier indexing

overrides = (;)
thermo_params, surf_flux_params = get_surf_flux_params(overrides)

ρ_alt = zeros(Z)
ρ_3 = zeros(Z)
for i in 1:Z
    sum = 0
    for j in 1:T
        ts = TD.PhaseEquil_pTq(thermo_params, p_data[i], temp_data[i, j], qt_data[i, j])
        ρ = air_density(thermo_params, ts)
        sum += ρ
    end
    sum /= T
    ρ_alt[i] = sum
end

ENV["GKSwstype"] = "nul"

plot(ρ_data)
plot!(ρ_alt)

png("images/test_plot1")

# h = P / (ρ g)
h_alt = (p_data .- p_data[1]) ./ (ρ_data .* 9.81)
h_alt = abs.(h_alt)

plot(h_alt)
plot!(z_data)

png("images/test_plot2")

# h = 