# Imports
using LinearAlgebra, Random
using Distributions, Plots

using Downloads
using NCDatasets
FT = Float64

# We extract the relevant data points for our pipeline.
max_z_index = 5 # since MOST allows data only in the surface layer
spin_up = 100
dict = Dict("profiles" => ("z", "t"), "reference" => ("z", ), "timeseries" => ("t", ))

@assert (!isfile("data/LES_all.nc"))
ds = NCDataset("data/LES_all.nc", "c")

function copy_info(new_groupname, data, groupname, varname)
    old_var = data.group[groupname][varname]
    new_var = defVar(ds.group[new_groupname], varname, FT, dict[groupname])
    new_var.attrib = old_var.attrib
    if (groupname == "profiles")
        new_var[:, :] = Array(old_var)[1:max_z_index, spin_up:end]
    elseif (groupname == "reference")
        new_var[:] = Array(old_var)[1:max_z_index]
    elseif (groupname == "timeseries")
        new_var[:] = Array(old_var)[spin_up:end]
    end
end

observables = [("profiles", "u_mean"), ("profiles", "v_mean"), ("profiles", "qt_mean"), 
               ("profiles", "thetali_mean"), ("profiles", "temperature_mean"),
               ("reference", "z"), ("reference", "rho0"), ("reference", "p0"),
               ("timeseries", "t"), ("timeseries", "friction_velocity_mean"),
               ("timeseries", "lhf_surface_mean"), ("timeseries", "shf_surface_mean"),
               ("timeseries", "surface_temperature"), ("timeseries", "obukhov_length_mean")]

months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

for cfSite in 17:23
    for month in months
        localfile = "data/Stats.cfsite$(cfSite)_CNRM-CM5_amip_2004-2008.$(month).nc"
        if (isfile(localfile))
            data = NCDataset(localfile)

            # add to new NCDataset
            groupname = "cfSite_$(cfSite)_month_$(month)"
            defGroup(ds, groupname)

            defDim(ds.group[groupname], "z", 5)
            defDim(ds.group[groupname], "t", 766)
            
            for (grp, var) in observables
                copy_info(groupname, data, grp, var)
            end
        end
    end
end

close(ds)