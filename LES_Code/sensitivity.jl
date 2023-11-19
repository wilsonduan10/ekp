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

ENV["GKSwstype"] = "nul"
include("../helper/setup_parameter_set.jl")
include("load_data.jl")
include("physical_model.jl")

# get data
cfSite = 23
month = 7

data = create_dataframe(cfSite, month)
Z, T = size(data.u)

outputdir = "images/LES_sensitivity"
for i in 1:length(outputdir)
    if (outputdir[i] == '/')
        mkpath(outputdir[1:i-1])
    end
end
mkpath(outputdir)

# other model parameters
parameterTypes = (:b_m, :b_h)
ufpt = UF.BusingerType()
phase_fn = ρTq()
scheme = ValuesOnlyScheme()
function H(output)
    observable = zeros(T)
    for j in 1:T
        sum = 0.0
        total = 0
        for i in 1:Z
            if (!isnothing(output[i, j]))
                sum += output[i, j].ustar
                total += 1
            end
        end
        observable[j] = sum / total
    end
    return observable
end

function G(parameters)
    Ψ = physical_model(parameters, parameterTypes, data, ufpt, phase_fn, scheme)
    return H(Ψ)
end

theta_true = (15.0, 9.0)

# b_m
b_ms = collect(6:3:21)
plot()
for b_m in b_ms
    ustars = G((b_m, 9.0))
    plot!(ustars, label=("b_m = " * string(b_m)))
end
xlabel!("Time")
ylabel!("ustar")
png("$(outputdir)/b_m")

# b_h
b_hs = collect(3:3:18)
plot()
for b_h in b_hs
    ustars = G((15.0, b_h))
    plot!(ustars, label=("b_h = " * string(b_h)))
end
xlabel!("Time")
ylabel!("ustar")
png("$(outputdir)/b_h")

# z0
old_z0 = data.z0
z0s = [0.01, 0.001, 0.0001, 0.00001]
plot()
for z0 in z0s
    data.z0 = z0
    ustars = G(theta_true)
    plot!(ustars, label=("z0 = " * string(z0)))
end
xlabel!("Time")
ylabel!("ustar")
png("$(outputdir)/z0")
data.z0 = old_z0

# T_sfc
old_T_sfc = data.T_sfc
T_sfcs = collect(280:5:310)
plot()
for T_sfc in T_sfcs
    data.T_sfc = fill(T_sfc, length(old_T_sfc))
    ustars = G(theta_true)
    plot!(ustars, label=("T_sfc = " * string(T_sfc)))
end
xlabel!("Time")
ylabel!("ustar")
png("$(outputdir)/T_sfc")
data.T_sfc = old_T_sfc

# qt_sfc
old_qt_sfc = data.qt_sfc
qt_sfcs = collect(0.01:0.003:0.02)
plot()
for qt_sfc in qt_sfcs
    data.qt_sfc = fill(qt_sfc, length(old_qt_sfc))
    ustars = G(theta_true)
    plot!(ustars, label=("qt_sfc = " * string(qt_sfc)))
end
xlabel!("Time")
ylabel!("ustar")
png("$(outputdir)/qt_sfc")
data.qt_sfc = old_qt_sfc

# ρ_sfc
old_ρ_sfc = data.ρ_sfc
ρ_sfcs = [0.01, 0.001, 0.0001, 0.00001]
plot()
for ρ_sfc in ρ_sfcs
    data.ρ_sfc = ρ_sfc
    ustars = G(theta_true)
    plot!(ustars, label=("ρ_sfc = " * string(ρ_sfc)))
end
xlabel!("Time")
ylabel!("ustar")
png("$(outputdir)/ρ_sfc")
data.ρ_sfc = old_ρ_sfc
