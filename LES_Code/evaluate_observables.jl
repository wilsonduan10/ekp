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
include("load_data.jl")
include("physical_model.jl")

# get data
cfSite = 23
month = 7

data = create_dataframe(cfSite, month)
Z, T = size(data.u)

outputdir = "images/LES_observables/observables_$(cfSite)_$(month)"
mkpath(outputdir)

# other model parameters
parameterTypes = (:b_m, :b_h)
ufpt = UF.BusingerType()
phase_fn = ρTq()
scheme = ValuesOnlyScheme()

function H(output)
    ustar = zeros(T)
    shf = zeros(T)
    lhf = zeros(T)
    buoy_flux = zeros(T)
    lmo = zeros(T)
    for j in 1:T
        sums = zeros(5)
        total = 0
        for i in 1:Z
            if (!isnothing(output[i, j]))
                sums[1] += output[i, j].ustar
                sums[2] += output[i, j].shf
                sums[3] += output[i, j].lhf
                sums[4] += output[i, j].buoy_flux
                sums[5] += output[i, j].L_MO
                total += 1
            end
        end
        ustar[j] = sums[1] / total
        shf[j] = sums[2] / total
        lhf[j] = sums[3] / total
        buoy_flux[j] = sums[4] / total
        lmo[j] = sums[5] / total
    end
    return ustar, shf, lhf, buoy_flux, lmo
end

function G(parameters)
    Ψ = physical_model(parameters, parameterTypes, data, ufpt, phase_fn, scheme)
    return H(Ψ)
end

theta_true = (15.0, 9.0)
ustar_truth, shf_truth, lhf_truth, buoy_flux_truth, LMO_truth = G(theta_true)

# plot ustar
plot(data.u_star, c=:green, seriestype=:scatter, ms=5, label="data")
plot!(ustar_truth, c=:red, seriestype=:scatter, ms=5, label="predicted")
title!("ustar comparison")
xlabel!("T")
ylabel!("ustar")
png("$(outputdir)/ustar_plot")

# plot shf
plot(data.shf, c=:green, seriestype=:scatter, ms=5, label="data")
plot!(shf_truth, c=:red, seriestype=:scatter, ms=5, label="predicted")
title!("shf comparison")
xlabel!("T")
ylabel!("shf")
png("$(outputdir)/shf_plot")

# plot lhf
plot(data.lhf, c=:green, seriestype=:scatter, ms=5, label="data")
plot!(lhf_truth, c=:red, seriestype=:scatter, ms=5, label="predicted")
title!("lhf comparison")
xlabel!("T")
ylabel!("lhf")
png("$(outputdir)/lhf_plot")

# buoy flux
plot(data.buoy_flux, c=:green, seriestype=:scatter, ms=5, label="data")
plot!(buoy_flux_truth, c=:red, seriestype=:scatter, ms=5, label="predicted")
title!("buoy_flux comparison")
xlabel!("T")
ylabel!("buoy_flux")
png("$(outputdir)/buoy_flux_plot")

# LMO
plot(data.L_MO, c=:green, seriestype=:scatter, ms=5, label="data")
plot!(LMO_truth, c=:red, seriestype=:scatter, ms=5, label="predicted")
title!("LMO comparison")
xlabel!("T")
ylabel!("LMO")
png("$(outputdir)/LMO_plot")