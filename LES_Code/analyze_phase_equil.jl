# Imports
using LinearAlgebra, Random
using Distributions, Plots

using Downloads
using NCDatasets

using CLIMAParameters
const CP = CLIMAParameters
FT = Float64

import SurfaceFluxes as SF
import SurfaceFluxes.Parameters as SFP
import Thermodynamics as TD

# We include some helper files. The first is to set up the parameters for surface\_conditions, and
# the second is to plot our results.
include("../helper/setup_parameter_set.jl")
include("../helper/graph.jl")

# We extract data from LES driven by GCM forcings, see https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002631.
# We must first download the netCDF datasets and place them into the data/ directory. We have the option to choose
# the cfsite and the month where data is taken from, as long as the data has been downloaded.
cfsite = 23
month = "01"
localfile = "data/Stats.cfsite$(cfsite)_CNRM-CM5_amip_2004-2008.$(month).nc"
data = NCDataset(localfile)

mkpath("images/PhaseEquil_plots")

# We extract the relevant data points for our pipeline.
max_z_index = 5 # since MOST allows data only in the surface layer
spin_up = 100

# profiles
u_data = Array(data.group["profiles"]["u_mean"])[1:max_z_index, spin_up:end]
v_data = Array(data.group["profiles"]["v_mean"])[1:max_z_index, spin_up:end]
qt_data = Array(data.group["profiles"]["qt_mean"])[1:max_z_index, spin_up:end]
θ_li_data = Array(data.group["profiles"]["thetali_mean"])[1:max_z_index, spin_up:end]
temp_data = Array(data.group["profiles"]["temperature_mean"])[1:max_z_index, spin_up:end]

# reference
z_data = Array(data.group["reference"]["z"])[1:max_z_index]
ρ_data = Array(data.group["reference"]["rho0"])[1:max_z_index]
p_data = Array(data.group["reference"]["p0"])[1:max_z_index]

# timeseries
time_data = Array(data.group["timeseries"]["t"])[spin_up:end]
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"])[spin_up:end]
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"])[spin_up:end]
shf_data = Array(data.group["timeseries"]["shf_surface_mean"])[spin_up:end]
surface_temp_data = Array(data.group["timeseries"]["surface_temperature"])[spin_up:end]

Z, T = size(u_data) # extract dimensions for easier indexing

# We combine the u and v velocities into a single number to facilitate analysis: u = √u^2 + v^2
for i in 1:Z
    u_data[i, :] = sqrt.(u_data[i, :] .* u_data[i, :] .+ v_data[i, :] .* v_data[i, :])
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

thermo_params, _ = get_surf_flux_params((;))

# check TD.PhaseEquil_ρθq
p_alt = zeros(Z, T)
T_alt = zeros(Z, T)
for j in 1:T
    for i in 1:Z
        ts_in = TD.PhaseEquil_ρθq(thermo_params, ρ_data[i], θ_li_data[i, j], qt_data[i, j])

        p_alt[i, j] = TD.air_pressure(thermo_params, ts_in)
        T_alt[i, j] = TD.air_temperature(thermo_params, ts_in)
    end
end
p_alt = vec(mean(p_alt, dims=2))
T_alt = vec(mean(T_alt, dims=2))
T_data = vec(mean(temp_data, dims=2))

plot(z_data, p_data, label="data")
plot!(z_data, p_alt, label="estimate")
title!("Pressure Comparison with ρθq")
xlabel!("Z")
ylabel!("Pressure (Pa)")
png("images/PhaseEquil_plots/pressure_ρθq")

plot(z_data, T_data, label="data")
plot!(z_data, T_alt, label="estimate")
title!("Temperature Comparison with ρθq")
xlabel!("Z")
ylabel!("Temperature (K)")
png("images/PhaseEquil_plots/temp_ρθq")

# check TD.PhaseEquil_pTq
ρ_alt = zeros(Z, T)
for j in 1:T
    for i in 1:Z
        ts_in = TD.PhaseEquil_pTq(thermo_params, p_data[i], temp_data[i, j], qt_data[i, j])

        ρ_alt[i, j] = TD.air_density(thermo_params, ts_in)
    end
end
ρ_alt = vec(mean(ρ_alt, dims=2))

plot(z_data, ρ_data, label="data")
plot!(z_data, ρ_alt, label="estimate")
title!("Air density Comparison with pTq")
xlabel!("Z")
ylabel!("Air density (kg/m^3)")
png("images/PhaseEquil_plots/rho_pTq")

# check D.PhaseEquil_ρTq
p_alt = zeros(Z, T)
for j in 1:T
    for i in 1:Z
        ts_in = TD.PhaseEquil_ρTq(thermo_params, ρ_data[i], temp_data[i, j], qt_data[i, j])

        p_alt[i, j] = TD.air_pressure(thermo_params, ts_in)
    end
end
p_alt = vec(mean(p_alt, dims=2))
plot(z_data, p_data, label="data")
plot!(z_data, p_alt, label="estimate")
title!("Pressure Comparison with ρTq")
xlabel!("Z")
ylabel!("Pressure (Pa)")
png("images/PhaseEquil_plots/pressure_ρTq")

# analyze extrapolated surface values
println(ρ_data)
println(vec(mean(qt_data, dims=2)))

include(joinpath(@__DIR__, "../helper/setup_parameter_set.jl"))

inputs = (; ρ_data, p_data, surface_temp_data, temp_data, θ_li_data, qt_data)
ρ_temp, q_temp = extrapolate_sfc_state(inputs)
