using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

mkpath(joinpath(@__DIR__, "../images"))

function next_folder_number(filename, cfsite, month)
    folders = filter(x -> startswith(x, "$(filename)_$(cfsite)_$(month)"), readdir(joinpath(@__DIR__, "../images")))
    string_length = length("$(filename)_$(cfsite)_$(month)")
    folder_nums = map(x -> parse(Int64, x[string_length+2:end]), folders)
    folder_number = 1
    if (length(folder_nums) > 0)
        folder_number = maximum(folder_nums) + 1
    end
    return folder_number
end

function next_SHEBA_number()
    folders = filter(x -> startswith(x, "SHEBA_"), readdir(joinpath(@__DIR__, "../images")))
    string_length = length("SHEBA_")
    folder_nums = map(x -> parse(Int64, x[string_length+1:end]), folders)
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
    (; axes, filename, folder_number) = kwargs
    plot(prior)
    png("images/$(filename_to_string(filename, folder_number))/prior_plot")
end

function plot_good_bad_model(x, model, model_truth, model_bad, inputs, kwargs)
    (; axes, filename, folder_number) = kwargs
    truth = model(model_truth, inputs)
    bad = model(model_bad, inputs)
    plot(x, truth, c = :black, label = "Model Truth", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    plot!(x, bad, c = :red, label = "Model Bad", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/$(filename_to_string(filename, folder_number))/good_bad_model")
end

function plot_noise(x, y, data, kwargs)
    (; axes, filename, folder_number) = kwargs
    plot(x, y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    plot!(x, data, c = :red, label = "Data Truth", ms = 1.5, seriestype=:scatter)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/$(filename_to_string(filename, folder_number))/y vs data")
end

function plot_y_versus_model(x, y, model, theta_true, inputs, kwargs)
    (; axes, filename, folder_number) = kwargs
    truth = model(theta_true, inputs)
    plot(x, y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter,)
    plot!(x, truth, c = :black, label = "Model Truth", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/$(filename_to_string(filename, folder_number))/good model and y")
end

function plot_all(x, y, model, theta_true, inputs, ensembles, N_ensemble, kwargs)
    (; axes, filename, folder_number) = kwargs
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
    png("images/$(filename_to_string(filename, folder_number))/our_plot")
end

function plot_z0s(x, y, z0s, model, theta_true, most_inputs, kwargs)
    (; axes, filename, folder_number) = kwargs
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
    png("images/$(filename_to_string(filename, folder_number))/z0_plot")
end

function generate_all_plots(params, filename, cfsite, month, new_folder = false)
    folder_number = 0
    if (new_folder)
        folder_number = next_folder_number(filename, cfsite, month)
    end
    mkpath(joinpath(@__DIR__, "../images/$(filename)_$(cfsite)_$(month)_$(folder_number)"))
    
    kwargs = (;
        axes = params.ax,
        filename = (filename, cfsite, month),
        folder_number = folder_number
    )

    # plot priors
    plot_prior(params.prior, kwargs)

    # plot good and bad model
    plot_good_bad_model(params.x, params.model, params.theta_true, params.theta_bad, params.inputs, kwargs)

    # plot good model and y
    plot_y_versus_model(params.x, params.y, params.model, params.theta_true, params.inputs, kwargs)

    # plot y, good model, and ensembles
    plot_all(params.x, params.y, params.model, params.theta_true, params.inputs, params.ensembles, params.N_ensemble, kwargs)

    # plot y versus model truth given different z0
    plot_z0s(params.x, params.y, params.z0s, params.model, params.theta_true, params.most_inputs, kwargs)

    println("Generated plots in folder: images/$(filename)_$(cfsite)_$(month)_$(folder_number)")
end

function generate_SHEBA_plots(params, new_folder = false)
    folder_number = 0
    if (new_folder)
        folder_number = next_SHEBA_number()
    end
    mkpath(joinpath(@__DIR__, "../images/SHEBA_$(folder_number)"))
    
    kwargs = (;
        axes = params.ax,
        filename = ("SHEBA", ),
        folder_number = folder_number
    )

    # plot priors
    plot_prior(params.prior, kwargs)

    # plot good and bad model
    plot_good_bad_model(params.x, params.model, params.theta_true, params.theta_bad, params.inputs, kwargs)

    # plot good model and y
    plot_y_versus_model(params.x, params.y, params.model, params.theta_true, params.inputs, kwargs)

    # plot y, good model, and ensembles
    plot_all(params.x, params.y, params.model, params.theta_true, params.inputs, params.ensembles, params.N_ensemble, kwargs)

    println("Generated plots in folder: images/SHEBA_$(folder_number)")
end