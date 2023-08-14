using LinearAlgebra, Random
using Distributions, Plots
using Downloads
using NCDatasets

mkpath(joinpath(@__DIR__, "data")) # create data folder if not exists
mkpath(joinpath(@__DIR__, "images"))
mkpath(joinpath(@__DIR__, "images/time_profiles"))
localfile = "data/Stats.cfsite17_CNRM-CM5_amip_2004-2008.10.nc"
data = NCDataset(localfile)

# Extract data
time_data = Array(data.group["timeseries"]["t"]) # (865, )
z_data = Array(data.group["reference"]["z"]) # (200, )
u_star_data = Array(data.group["timeseries"]["friction_velocity_mean"]) # (865, )
u_data = Array(data.group["profiles"]["u_mean"]) # (200, 865)
v_data = Array(data.group["profiles"]["v_mean"]) # (200, 865)
ρ_data = Array(data.group["reference"]["rho0"]) # (200, )
qt_data = Array(data.group["profiles"]["qt_mean"]) # (200, 865)
θ_li_data = Array(data.group["profiles"]["thetali_mean"]) # (200, 865)
lhf_data = Array(data.group["timeseries"]["lhf_surface_mean"]) # (865, )
shf_data = Array(data.group["timeseries"]["shf_surface_mean"]) # (865, )
uw_data = Array(data.group["timeseries"]["uw_surface_mean"]) # (865, )
vw_data = Array(data.group["timeseries"]["vw_surface_mean"]) # (865, )

Z, T = size(u_data)
ENV["GKSwstype"] = "nul"

# plot mean u and v vs time
plot(time_data, [mean(u_data[:, i]) for i in 1:T], label="u")
plot!(time_data, [mean(v_data[:, i]) for i in 1:T], label="v")
xlabel!("Time")
ylabel!("Mean velocity")
png("images/time_profiles/u and v")

# plot uw and uv
plot(time_data, uw_data, label="uw")
plot!(time_data, vw_data, label="vw")
xlabel!("Time")
ylabel!("Flux")
png("images/time_profiles/uw and vw")

# plot lhf and shf vs time
plot(time_data, lhf_data, label="lhf")
plot!(time_data, shf_data, label="shf")
xlabel!("Time")
ylabel!("Surface heat flux")
png("images/time_profiles/surface heat flux")

# other plots todo:
# plot qt vs time
# plot buoyancy vs time
# plot theta vs time
# plot ustar vs time