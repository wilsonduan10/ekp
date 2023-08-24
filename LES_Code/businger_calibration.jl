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

FT = Float64

import RootSolvers
const RS = RootSolvers

import SurfaceFluxes as SF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFP
using StaticArrays: SVector

include("../helper/setup_parameter_set.jl")
include("../helper/graph.jl")
include("load_data.jl")
include("physical_model.jl")

# constants
cfSite = 23
month = 7
parameterTypes = (:b_m, :b_h)
ufpt = UF.BusingerType()
phase_fn = ρTq()
scheme = ValuesOnlyScheme()
outputdir = "images/businger_calibration/bc_$(cfSite)_$(month)_0"
mkpath(outputdir)

data = create_dataframe(cfSite, month)

# Our function G simply returns the output of the physical model.
function G(parameters)
    return physical_model(parameters, parameterTypes, data, ufpt, phase_fn, scheme)
end

y = data.u_star
variance = 0.05^2 * (maximum(y) - minimum(y)) # assume 5% noise
Γ = variance * I

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
    G_ens = hcat([G(params_i[:, m]) for m in 1:N_ensemble]...)
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
    err = get_error(ensemble_kalman_process)[end] #mean((params_true - mean(params_i,dims=2)).^2)
    println("Iteration: " * string(n) * ", Error: " * string(err))
end

# We extract the constrained initial and final ensemble for analysis
constrained_initial_ensemble = get_ϕ(prior, ensemble_kalman_process, 1)
final_ensemble = get_ϕ_final(prior, ensemble_kalman_process)

# We print the mean parameters of the initial and final ensemble to identify how
# the parameters evolved to fit the dataset. 
println("\nINITIAL ENSEMBLE STATISTICS")
println("Mean b_m:", mean(constrained_initial_ensemble[1, :]))
println("Mean b_h:", mean(constrained_initial_ensemble[2, :]))
println()

println("FINAL ENSEMBLE STATISTICS")
println("Mean b_m:", mean(final_ensemble[1, :]))
println("Mean b_h:", mean(final_ensemble[2, :]))

theta_true = (15.0, 9.0)
model_truth = G(theta_true)
plot(model_truth, label="Model Truth", c=:red, seriestype=:scatter, ms=5)
plot!(y, label="y", c=:green, seriestype=:scatter, ms=5)
xlabel!("T")
ylabel!("ustar")
png("$(outputdir)/test_plot")

initial = [G(constrained_initial_ensemble[:, i]) for i in 1:N_ensemble]
final = [G(final_ensemble[:, i]) for i in 1:N_ensemble]
initial_label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
final_label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)

plot(data.time, y, c = :green, label = "y", legend = :bottomright, ms = 5, seriestype=:scatter, markerstroke="green", markershape=:utriangle)
plot!(data.time, initial, c = :red, label = initial_label, linewidth = 2)
plot!(data.time, final, c = :blue, label = final_label, linewidth = 2)
xlabel!("T")
ylabel!("ustar")
png("$(outputdir)/test_plot2")