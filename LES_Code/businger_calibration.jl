#=
In this file, we use the ensemble kalman inversion process to calibrate four parameters:
a\_m, a\_h, b\_m, and b\_h in the Businger stability functions, where a_m and a_h are parameters
in the stable regime, and b_m and b_h pertain to the unstable regime. We use data from Shen 2022,
cfSite LES data with GCM forcings. We use u_star as our observable, and predict u_star using 
the function surface_conditions from the SurfaceFluxes.jl package, which uses the Monin-Obukhov
similarity theory to calculate important scales such as the monin-obukhov length and u_star. The 
goal of this process is to analyze the efficacy of EKP in the context of surface fluxes. We visualize
the success of the model through several plots.
=#

# Imports
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

# We include some helper files. The first is to set up the parameters for surface\_conditions, and
# the second is to plot our results.
include("../helper/setup_parameter_set.jl")
include("../helper/graph.jl")

# We extract data from LES driven by GCM forcings, see https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002631.
# We must first download the netCDF datasets and place them into the data/ directory. We have the option to choose
# the cfsite and the month where data is taken from, as long as the data has been downloaded.
cfsite = 20
month = "07"
localfile = "data/Stats.cfsite$(cfsite)_CNRM-CM5_amip_2004-2008.$(month).nc"
data = NCDataset(localfile)

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

# Because the model sometimes fails to converge, we store unconverged values in a dictionary
# so we can analyze and uncover the cause of failure.
unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()

# We define our physical model. It takes in the parameters a\_m, a\_h, b\_m, b\_h, as well as data
# inputs. It establishes thermodynamic parameters and Businger parameters in order to call the 
# function surface_conditions. We store each time step's u_star and return a list of these u_stars.
function physical_model(parameters, inputs)
    b_m, b_h = parameters
    (; u, z, time, lhf, shf, z0) = inputs

    overrides = (; b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides) # override default Businger params

    # Now, we loop over all the observations and call SF.surface_conditions to estimate u_star
    output = zeros(T)
    for j in 1:T
        sum = 0.0
        total = 0
        # Establish surface conditions
        # ts_sfc = TD.PhaseEquil_ρθq(thermo_params, ρ_data[1], θ_li_data[1, j], qt_data[1, j]) # use 1 to get surface conditions
        ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_data[1], surface_temp_data[j], qt_data[1, j])
        # ts_sfc = TD.PhaseEquil_pTq(thermo_params, p_data[1], temp_data[1, j], qt_data[1, j])
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        # u_sfc = SVector{2, FT}(u[1, j], FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)
        # state_sfc = SF.SurfaceValues(z[1], u_sfc, ts_sfc)

        # We now loop through all heights at this time step.
        for i in 2:Z
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i]
            u_in = SVector{2, FT}(u_in, v_in)
            
            # ts_in = TD.PhaseEquil_ρθq(thermo_params, ρ_data[i], θ_li_data[i, j], qt_data[i, j])
            ts_in = TD.PhaseEquil_pTq(thermo_params, p_data[i], temp_data[i, j], qt_data[i, j])
            state_in = SF.InteriorValues(z_in, u_in, ts_in)

            # We provide a few additional parameters for SF.surface_conditions
            z0m = z0b = z0
            gustiness = FT(1)
            # kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
            # sc = SF.Fluxes{FT}(; kwargs...)
            kwargs = (state_in = state_in, state_sfc = state_sfc, z0m = z0m, z0b = z0b, gustiness = gustiness)
            sc = SF.ValuesOnly{FT}(; kwargs...)

            # Now, we call surface_conditions and store the calculated ustar. We surround it in a try catch
            # to account for unconverged fluxes.
            try
                sf = SF.surface_conditions(surf_flux_params, sc, soltype = RS.VerboseSolution())
                sum += sf.ustar
                total += 1
            catch e
                # println(e)
                z_temp, t_temp = (z_data[i], time_data[j])
                temp_key = (z_temp, t_temp)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end
            
        end
        output[j] = sum / total
    end
    return output
end

# Our function G simply returns the output of the physical model.
function G(parameters, inputs)
    return physical_model(parameters, inputs)
end

inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data, z0 = 0.0001)

variance = 0.05^2 * (maximum(u_star_data) - minimum(u_star_data)) # assume 5% noise
Γ = variance * I
y = u_star_data

# Define the prior parameter values which we wish to recover in our pipeline. They are constrained
# to be non-negative due to physical laws, and their mean is given by Businger et al 1971.
prior_u1 = constrained_gaussian("b_m", 15.0, 8, 0, 30)
prior_u2 = constrained_gaussian("b_h", 9.0, 6, 0, 30)
prior = combine_distributions([prior_u1, prior_u2])

N_ensemble = 5
N_iterations = 5

# Define EKP process.
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)

# Run EKI for N_iterations
for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
    err = get_error(ensemble_kalman_process)[end] #mean((params_true - mean(params_i,dims=2)).^2)
    println("Iteration: " * string(n) * ", Error: " * string(err))
end

# We extract the constrained initial and final ensemble for analysis
constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

# Print the unconverged data points to identify a pattern.
if (length(unconverged_data) > 0)
    println("Unconverged data points: ", unconverged_data)
    println("Unconverged z: ", unconverged_z)
    println("Unconverged t: ", unconverged_t)
    println()
end

# We print the mean parameters of the initial and final ensemble to identify how
# the parameters evolved to fit the dataset. 
println("\nINITIAL ENSEMBLE STATISTICS")
println("Mean b_m:", mean(constrained_initial_ensemble[1, :]))
println("Mean b_h:", mean(constrained_initial_ensemble[2, :]))
println()

println("FINAL ENSEMBLE STATISTICS")
println("Mean b_m:", mean(final_ensemble[1, :]))
println("Mean b_h:", mean(final_ensemble[2, :]))

# In order to plot, we define a set of parameters:
plot_params = (;
    x = time_data,
    y = y,
    ax = ("T", "U*"),
    prior = prior,
    model = physical_model,
    inputs = inputs,
    theta_true = (15.0, 9.0),
    ensembles = (constrained_initial_ensemble, final_ensemble),
    N_ensemble = N_ensemble
)

println()
generate_all_plots(plot_params, "businger_calibration", "bc", cfsite, month, false)
