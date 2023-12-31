# how I generated the plots used in the google slides presentation
using LinearAlgebra, Random
using Distributions, Plots

z0 = 0.1
z = z0:1:100
ustar = 0.27
theta_star = -0.2
theta_sfc = 298
u_profile = ustar / 0.4 .* log.(z ./ z0)
theta_profile = theta_sfc .+ theta_star / 0.4 .* log.(z ./ z0)

plot(u_profile, z, label="", linewidth=6)
xlabel!("u (m/s)")
ylabel!("z (m)")
title!("Wind Speed (u) vs Altitude (z)")
png("u profile")

plot(theta_profile, z, label="", linewidth=6)
xlabel!("θ (K)")
ylabel!("z (m)")
title!("Potential Temperature (θ) vs Altitude (z)")
png("θ profile")