using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

mkpath(joinpath(@__DIR__, "../images"))

function next_bc_number(cfsite)
    bc_folders = filter(x -> startswith(x, "bc_$cfsite"), readdir(joinpath(@__DIR__, "../images")))
    bc_nums = map(x -> parse(Int64, x[7:end]), bc_folders)
    bc_number = 1
    if (length(bc_nums) > 0)
        bc_number = maximum(bc_nums) + 1
    end
    mkpath(joinpath(@__DIR__, "../images/bc$bc_number"))
    return bc_number
end

ENV["GKSwstype"] = "nul"
theta_true = (4.7, 4.7, 15.0, 9.0)

function plot_prior(prior, kwargs)
    (; axes, cfsite, bc_number) = kwargs
    plot(prior)
    png("images/bc_$(cfsite)_$(bc_number)/prior_plot")
end

function plot_good_bad_model(x, model, model_truth, model_bad, inputs, kwargs)
    (; axes, cfsite, bc_number) = kwargs
    truth = model(model_truth, inputs)
    bad = model(model_bad, inputs)
    plot(x, truth, c = :black, label = "Model Truth", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    plot!(x, bad, c = :red, label = "Model Bad", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/bc_$(cfsite)_$(bc_number)/good_bad_model")
end

function plot_noise(x, y, data, kwargs)
    (; axes, cfsite, bc_number) = kwargs
    plot(x, y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    plot!(x, data, c = :red, label = "Data Truth", ms = 1.5, seriestype=:scatter)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/bc_$(cfsite)_$(bc_number)/y vs data")
end

function plot_y_versus_model(x, y, model, theta_true, inputs, kwargs)
    (; axes, cfsite, bc_number) = kwargs
    truth = model(theta_true, inputs)
    plot(x, y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter,)
    plot!(x, truth, c = :black, label = "Model Truth", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/bc_$(cfsite)_$(bc_number)/good model and y")
end

function plot_all(x, y, model, theta_true, inputs, ensembles, N_ensemble, kwargs)
    (; axes, cfsite, bc_number) = kwargs
    initial_ensemble, final_ensemble = ensembles
    initial = [model(initial_ensemble[:, i], inputs) for i in 1:N_ensemble]
    final = [model(final_ensemble[:, i], inputs) for i in 1:N_ensemble]
    initial_label = reshape(vcat(["Initial ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)
    final_label = reshape(vcat(["Final ensemble"], ["" for i in 1:(N_ensemble - 1)]), 1, N_ensemble)

    plot(x, y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter,)
    plot!(x, model(theta_true, inputs), c = :black, label = "Model Truth", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    plot!(x, initial, c = :red, label = initial_label)
    plot!(x, final, c = :blue, label = final_label)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/bc_$(cfsite)_$(bc_number)/our_plot")
end

function plot_z0s(x, y, z0s, model, theta_true, most_inputs, kwargs)
    (; axes, cfsite, bc_number) = kwargs
    plot(x, y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter)

    for i in 1:length(z0s)
        custom_input = (; most_inputs..., z0 = z0s[i])
        output = model(theta_true, custom_input)
        plot!(time_data, output, label="z0 = " * string(z0s[i]))
    end

    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/bc_$(cfsite)_$(bc_number)/z0_plot")
end

function generate_bc_plots(params, cfsite, new_folder = false)
    bc_number = 0
    if (new_folder)
        bc_number = next_bc_number(cfsite)
    end
    mkpath(joinpath(@__DIR__, "../images/bc_$(cfsite)_$(bc_number)"))
    
    kwargs = (;
        axes = params.ax,
        cfsite = cfsite,
        bc_number = bc_number
    )

    # plot priors
    plot_prior(params.prior, kwargs)

    # plot good and bad model
    plot_good_bad_model(params.x, params.model, params.theta_true, params.theta_bad, params.inputs, kwargs)

    # plot y vs u_star data
    u_star_data = params.observable
    plot_noise(params.x, params.y, u_star_data, kwargs)

    # plot good model and y
    plot_y_versus_model(params.x, params.y, params.model, params.theta_true, params.inputs, kwargs)

    # plot y, good model, and ensembles
    plot_all(params.x, params.y, params.model, params.theta_true, params.inputs, params.ensembles, params.N_ensemble, kwargs)

    # plot y versus model truth given different z0
    plot_z0s(params.x, params.y, params.z0s, params.model, params.theta_true, params.most_inputs, kwargs)
end