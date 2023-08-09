# This file uses ψ as an observable, calculated from data metrics. The model is just the Businger ψ equation
using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

using Downloads
using NCDatasets
using CSV, DataFrames

using CLIMAParameters
const CP = CLIMAParameters
FT = Float64

import SurfaceFluxes as SF
import Thermodynamics as TD
import Thermodynamics.Parameters as TP
import SurfaceFluxes.UniversalFunctions as UF
import SurfaceFluxes.Parameters as SFP
using StaticArrays: SVector

include("../helper/setup_parameter_set.jl")

include("5level_data.jl")
L_MO_data = Matrix(CSV.read("data/L_MO.csv", DataFrame, header=false, delim='\t'))

# our model is ψ(z0m / L_MO) - ψ(z / L_MO)
function model(parameters, inputs)
    a_m, a_h, b_m, b_h = parameters
    (; z, L_MO) = inputs
    overrides = (; a_m, a_h, b_m, b_h)
    _, surf_flux_params = get_surf_flux_params(overrides)

    uft = UF.BusingerType()
    transport = UF.MomentumTransport()

    output = zeros(Z, T)
    for i in 1:Z
        for j in 1:T
            uf = UF.universal_func(uft, L_MO[i, j], SFP.uf_params(surf_flux_params))
            output[i, j] = UF.psi(uf, z0m / L_MO[i, j], transport)

            uf = UF.universal_func(uft, L_MO[i, j], SFP.uf_params(surf_flux_params))
            ζ = z[i, j] / L_MO[i, j]
            output[i, j] -= UF.psi(uf, ζ, transport)
        end
    end
    return vec(reshape(output, Z*T))
end

function G(parameters, inputs)
    return model(parameters, inputs)
end

# construct observable
z0m = 0.0001
κ = 0.4
y = κ * u_data ./ u_star_data - log.(z_data ./ z0m)

plot(reshape(ζ_data, Z*T), reshape(y, Z*T), seriestype=:scatter)
xlabel!("ζ")
ylabel!("ψ")

mkpath("images/SHEBA_psi")
png("images/SHEBA_psi/test_plot")
