```@meta
EditURL = "<unknown>/businger_calibration.jl"
```

In this file, we calibrate four parameters: a\_m, a\_h, b\_m, and b\_h in the Businger
stability functions, where a_m and a_h are parameters in the stable regime, and b_m and
b_h pertain to the unstable regime.

Imports

````@example businger_calibration
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
````

We include some helper files. The first is to set up the parameters for surface\_conditions, and
the second is to plot our results.

````@example businger_calibration
include("helper/setup_parameter_set.jl")
include("helper/graph.jl")
````

We extract data from LES driven by GCM forcings, see https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002631.
We must first download the netCDF datasets and place them into the data/ directory. We have the option to choose
the cfsite and the month where data is taken from, as long as the data has been downloaded.

````@example businger_calibration
mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
cfsite = 23
month = "01"
localfile = "data/Stats.cfsite$(cfsite)_CNRM-CM5_amip_2004-2008.$(month).nc"
data = NCDataset(localfile)
````

We extract the relevant data points for our pipeline.

````@example businger_calibration
time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["profiles"]["z"]) # (200, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"]) # (865, )
u_data = Array(data.group["profiles"]["u_mean"]) # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"]) # (200, 865)
ρ_data = Array(data.group["reference"]["rho0"]) # (200, )
qt_data = Array(data.group["profiles"]["qt_min"]) # (200, 865)
θ_li_data = Array(data.group["profiles"]["thetali_mean"]) # (200, 865)
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"]) # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"]) # (865, )

Z, T = size(u_data) # extract dimensions for easier indexing
````

We combine the u and v velocities into a single number to facilitate analysis: u = √u^2 + v^2

````@example businger_calibration
for i in 1:Z
    for j in 1:T
        u_data[i, j] = sqrt(u_data[i, j] * u_data[i, j] + v_data[i, j] * v_data[i, j])
    end
end
````

Because the model sometimes fails to converge, we store unconverged values in a dictionary
so we can analyze and uncover the cause of failure.

````@example businger_calibration
unconverged_data = Dict{Tuple{FT, FT}, Int64}()
unconverged_z = Dict{FT, Int64}()
unconverged_t = Dict{FT, Int64}()
````

We define our physical model. It takes in the parameters a\_m, a\_h, b\_m, b\_h, as well as data
inputs. It establishes thermodynamic parameters and Businger parameters in order to call the
function surface_conditions. We store each time step's u_star and return a list of these u_stars.

````@example businger_calibration
function physical_model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; u, z, time, lhf, shf, z0) = inputs

    overrides = (; a_m, a_h, b_m, b_h)
    thermo_params, surf_flux_params = get_surf_flux_params(overrides) # override default Businger params
````

Now, we loop over all the observations and call SF.surface_conditions to estimate u_star

````@example businger_calibration
    u_star = zeros(length(time)) # (865, )
    for j in 1:T # 865
        u_star_sum = 0.0
        total = 0
````

Define surface conditions based on moist air density, liquid ice potential temperature, and total specific humidity
given from cfSite.

````@example businger_calibration
        ts_sfc = TD.PhaseEquil_ρθq(thermo_params, ρ_data[1], θ_li_data[1, j], qt_data[1, j]) # use 1 to get surface conditions
        u_sfc = SVector{2, FT}(FT(0), FT(0))
        state_sfc = SF.SurfaceValues(FT(0), u_sfc, ts_sfc)
````

We now loop through all heights at this time step.

````@example businger_calibration
        for i in 2:Z # starting at 2 because index 1 is our surface conditions
            u_in = u[i, j]
            v_in = FT(0)
            z_in = z[i]
            u_in = SVector{2, FT}(u_in, v_in)

            ts_in = TD.PhaseEquil_ρθq(thermo_params, ρ_data[i], θ_li_data[i, j], qt_data[i, j])
            state_in = SF.InteriorValues(z_in, u_in, ts_in)
````

We provide a few additional parameters for SF.surface_conditions

````@example businger_calibration
            z0m = z0b = z0
            gustiness = FT(1)
            kwargs = (state_in = state_in, state_sfc = state_sfc, shf = shf[j], lhf = lhf[j], z0m = z0m, z0b = z0b, gustiness = gustiness)
            sc = SF.Fluxes{FT}(; kwargs...)
````

Now, we call surface_conditions and store the calculated ustar. We surround it in a try catch
to account for unconverged fluxes.

````@example businger_calibration
            try
                sf = SF.surface_conditions(surf_flux_params, sc, soltype=RS.VerboseSolution())
                u_star_sum += sf.ustar
                total += 1
            catch e
                println(e)

                z_temp, t_temp = (z_data[i], time_data[j])
                temp_key = (z_temp, t_temp)
                haskey(unconverged_data, temp_key) ? unconverged_data[temp_key] += 1 : unconverged_data[temp_key] = 1
                haskey(unconverged_z, z_temp) ? unconverged_z[z_temp] += 1 : unconverged_z[z_temp] = 1
                haskey(unconverged_t, t_temp) ? unconverged_t[t_temp] += 1 : unconverged_t[t_temp] = 1
            end

        end
````

We average the ustar over all heights and store it.

````@example businger_calibration
        total = max(total, 1) # just in case total is zero, we don't want to divide by 0
        u_star[j] = u_star_sum / total
    end
    return u_star
end
````

Our function G simply returns the output of the physical model.

````@example businger_calibration
function G(parameters, inputs)
    u_star = physical_model(parameters, inputs) # (865, )
    return u_star
end
````

Define inputs based on data, to be fed into the physical model.

````@example businger_calibration
inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data, z0 = 0.0001)
````

We define the noise parameter η, and add normally distributed noise to the given u_star_data
in order to define our observable y.

````@example businger_calibration
Γ = 0.00005 * I
η_dist = MvNormal(zeros(length(u_star_data)), Γ)
y = u_star_data .+ rand(η_dist) # (H ⊙ Ψ ⊙ T^{-1})(θ) + η from Cleary et al 2021
````

Define the prior parameter values which we wish to recover in our pipeline. They are constrained
to be non-negative due to physical laws, and their mean is given by Businger et al 1971.

````@example businger_calibration
prior_u1 = constrained_gaussian("a_m", 4.7, 3, 0, Inf);
prior_u2 = constrained_gaussian("a_h", 4.7, 3, 0, Inf);
prior_u3 = constrained_gaussian("b_m", 15.0, 8, 0, Inf);
prior_u4 = constrained_gaussian("b_h", 9.0, 6, 0, Inf);
prior = combine_distributions([prior_u1, prior_u2, prior_u3, prior_u4])
````

Hyperparameters: we find it sufficient to define just 5 ensembles and iterations.

````@example businger_calibration
N_ensemble = 5;
N_iterations = 5;
nothing #hide
````

Define EKP process.

````@example businger_calibration
rng_seed = 41
rng = Random.MersenneTwister(rng_seed)
initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble);

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng)
````

We run ensemble kalman inversrion for N_iterations

````@example businger_calibration
for n in 1:N_iterations
    params_i = get_ϕ_final(prior, ensemble_kalman_process)
    G_ens = hcat([G(params_i[:, m], inputs) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end
````

We extract the constrained initial and final ensemble for analysis.

````@example businger_calibration
constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)
````

In order to plot, we define a set of parameters:

````@example businger_calibration
plot_params = (;
    x = time_data,
    y = y,
    observable = u_star_data,
    ax = ("T", "U*"),
    prior = prior,
    model = physical_model,
    inputs = inputs,
    theta_true = (4.7, 4.7, 15.0, 9.0),
    theta_bad = (100.0, 100.0, 100.0, 100.0),
    ensembles = (constrained_initial_ensemble, final_ensemble),
    N_ensemble = N_ensemble,
    most_inputs = (u = u_data, z = z_data, time = time_data, lhf = lhf_data, shf = shf_data),
    z0s = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
)

generate_bc_plots(plot_params, cfsite, month, true)
````

Print the unconverged data points to identify a pattern.

````@example businger_calibration
if (length(unconverged_data) > 0)
    println("Unconverged data points: ", unconverged_data)
    println("Unconverged z: ", unconverged_z)
    println("Unconverged t: ", unconverged_t)
    println()
end
````

We print the mean parameters of the initial and final ensemble to identify how
the parameters evolved to fit the dataset.

````@example businger_calibration
println("INITIAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(constrained_initial_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(constrained_initial_ensemble[2, :]))
println("Mean b_m:", mean(constrained_initial_ensemble[3, :]))
println("Mean b_h:", mean(constrained_initial_ensemble[4, :]))
println()

println("FINAL ENSEMBLE STATISTICS")
println("Mean a_m:", mean(final_ensemble[1, :])) # [param, ens_no]
println("Mean a_h:", mean(final_ensemble[2, :]))
println("Mean b_m:", mean(final_ensemble[3, :]))
println("Mean b_h:", mean(final_ensemble[4, :]))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
