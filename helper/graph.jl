using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

mkpath(joinpath(@__DIR__, "../images"))

function next_folder_number(folder_name, filename, cfsite, month)
    folders = filter(x -> startswith(x, "$(filename)_$(cfsite)_$(month)"), readdir(joinpath(@__DIR__, "../images/$(folder_name)")))
    string_length = length("$(filename)_$(cfsite)_$(month)")
    folder_nums = map(x -> parse(Int64, x[string_length+2:end]), folders)
    folder_number = 1
    if (length(folder_nums) > 0)
        folder_number = maximum(folder_nums) + 1
    end
    return folder_number
end

function next_SHEBA_number()
    folders = filter(x -> startswith(x, "SHEBA_"), readdir(joinpath(@__DIR__, "../images/SHEBA")))
    string_length = length("SHEBA")
    folder_nums = map(x -> parse(Int64, x[string_length+2:end]), folders)
    folder_number = 1
    if (length(folder_nums) > 0)
        folder_number = maximum(folder_nums) + 1
    end
    return folder_number
end

ENV["GKSwstype"] = "nul"
theta_true = (4.7, 4.7, 15.0, 9.0)

function filename_to_string(filename, folder_number)
    output = ""
    for str in filename
        output *= string(str) * "_"
    end
    output *= string(folder_number)
    return output
end

function plot_prior(prior, kwargs)
    (; axes, folder_name, filename, folder_number) = kwargs
    plot(prior)
    png("images/$(folder_name)/$(filename_to_string(filename, folder_number))/prior_plot")
end

function plot_y_versus_model(x, y, model, theta_true, inputs, kwargs)
    (; axes, folder_name, filename, folder_number) = kwargs
    truth = model(theta_true, inputs)
    plot(x, y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter,)
    plot!(x, truth, c = :red, label = "Model Truth", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/$(folder_name)/$(filename_to_string(filename, folder_number))/good model and y")
end

function plot_histogram(y, model, theta_true, inputs, observable, name, kwargs)
    (; axes, folder_name, filename, folder_number) = kwargs
    truth = model(theta_true, inputs)
    plot(truth, y, c = :green, label = "", ms = 1.5, seriestype=:scatter,)
    
    # create dash: y = x
    x = minimum(y):0.01:maximum(y)
    plot!(x, x, c=:black, linestyle=:dash, linewidth=3, seriestype=:path)
    xlabel!("$(name) Predicted $(observable)")
    ylabel!("Observed $(observable)")
    title!("y vs $(name)")
    png("images/$(folder_name)/$(filename_to_string(filename, folder_number))/$(name)_2d_histogram")
end

function plot_all(x, y, model, theta_true, inputs, ensembles, N_ensemble, kwargs)
    (; axes, folder_name, filename, folder_number) = kwargs
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
    png("images/$(folder_name)/$(filename_to_string(filename, folder_number))/our_plot")
end

function plot_initial_mean(x, y, model, initial_mean, inputs, kwargs)
    (; axes, folder_name, filename, folder_number) = kwargs
    initial = model(initial_mean, inputs)

    plot(x, y, c = :green, label="y",  legend = :bottomright, ms = 1.5, seriestype=:scatter)
    plot!(x, initial, c=:red, label = "Mean Initial Ensemble", seriestype=:scatter, ms = 1.5)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/$(folder_name)/$(filename_to_string(filename, folder_number))/initial_ensemble")
end

function plot_final_mean(x, y, model, final_mean, inputs, kwargs)
    (; axes, folder_name, filename, folder_number) = kwargs
    final = model(final_mean, inputs)

    plot(x, y, c = :green, label="y",  legend = :bottomright, ms = 1.5, seriestype=:scatter)
    plot!(x, final, c=:blue, label = "Mean Final Ensemble", seriestype=:scatter, ms = 1.5)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/$(folder_name)/$(filename_to_string(filename, folder_number))/final_ensemble")
end

function plot_z0s(x, y, z0s, model, theta_true, most_inputs, kwargs)
    (; axes, folder_name, filename, folder_number) = kwargs
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
    png("images/$(folder_name)/$(filename_to_string(filename, folder_number))/z0_plot")
end

function generate_all_plots(params, folder_name, filename, cfsite, month, new_folder = false)
    folder_number = 0
    if (new_folder)
        folder_number = next_folder_number(folder_name, filename, cfsite, month)
    end
    mkpath(joinpath(@__DIR__, "../images/$(folder_name)/$(filename)_$(cfsite)_$(month)_$(folder_number)"))
    
    kwargs = (;
        axes = params.ax,
        folder_name, 
        filename = (filename, cfsite, month),
        folder_number = folder_number
    )

    # plot priors
    plot_prior(params.prior, kwargs)

    # plot good model and y
    plot_y_versus_model(params.x, params.y, params.model, params.theta_true, params.inputs, kwargs)

    # plot 2d_histogram
    plot_histogram(params.y, params.model, params.theta_true, params.inputs, "ustar", "Truth", kwargs)

    # plot y, good model, and ensembles
    plot_all(params.x, params.y, params.model, params.theta_true, params.inputs, params.ensembles, params.N_ensemble, kwargs)

    # plot y versus model truth given different z0
    plot_z0s(params.x, params.y, params.z0s, params.model, params.theta_true, params.most_inputs, kwargs)

    println("Generated plots in folder: images/$(folder_name)/$(filename)_$(cfsite)_$(month)_$(folder_number)")
end

function generate_SHEBA_plots(params, new_folder = false)
    folder_number = 0
    if (new_folder)
        folder_number = next_SHEBA_number()
    end
    mkpath(joinpath(@__DIR__, "../images/SHEBA/SHEBA_$(folder_number)"))
    
    kwargs = (;
        axes = params.ax,
        folder_name = "SHEBA",
        filename = ("SHEBA", ),
        folder_number = folder_number
    )

    # plot priors
    plot_prior(params.prior, kwargs)

    # plot good model and y
    plot_y_versus_model(params.x, params.y, params.model, params.theta_true, params.inputs, kwargs)

    # plot 2d_histogram
    plot_histogram(params.y, params.model, params.theta_true, params.inputs, "ustar", "Truth", kwargs)

    initial_ensemble, final_ensemble = params.ensembles
    # plot mean initial ensemble
    initial_mean = vec(mean(initial_ensemble, dims=2))
    plot_initial_mean(params.x, params.y, params.model, initial_mean, params.inputs, kwargs)

    # plot mean final ensemble
    final_mean = vec(mean(final_ensemble, dims=2))
    plot_final_mean(params.x, params.y, params.model, final_mean, params.inputs, kwargs)

    # plot initial ensemble vs y 2d_histogram
    plot_histogram(params.y, params.model, initial_mean, params.inputs, "ustar", "Initial Ensemble", kwargs)

    # plot final ensemble vs y 2d_histogram
    plot_histogram(params.y, params.model, final_mean, params.inputs, "ustar", "Final Ensemble", kwargs)

    println("Generated plots in folder: images/SHEBA/SHEBA_$(folder_number)")
end