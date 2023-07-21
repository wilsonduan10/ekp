using LinearAlgebra, Random
using Distributions, Plots
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

ENV["GKSwstype"] = "nul"
theta_true = (4.7, 4.7, 15.0, 9.0)

function plot_prior(prior)
    plot(prior)
    png("images/prior_plot")
end

function plot_good_bad_model(x, model_truth, model_bad, axes=nothing)
    plot(x, model_truth, c = :black, label = "Model Truth", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    plot!(x, model_bad, c = :red, label = "Model Bad", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/good_bad_model")
end

function plot_noise(x, y, data, axes=nothing)
    plot(x, y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    plot!(x, data, c = :red, label = "Data Truth", ms = 1.5, seriestype=:scatter)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/y vs data")
end

function plot_y_versus_model(x, y, model, axes=nothing)
    plot(x, y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter,)
    plot!(x, model, c = :black, label = "Model Truth", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/good model and y")
end

function plot_all(x, y, model_truth, initial, final, labels=("Initial ensemble", "Final ensemble"), axes=nothing)
    plot(x, y, c = :green, label = "y", legend = :bottomright, ms = 1.5, seriestype=:scatter,)
    plot!(x, model_truth, c = :black, label = "Model Truth", legend = :bottomright, ms = 1.5, seriestype=:scatter)
    plot!(x, initial, c = :red, label = labels[1])
    plot!(x, final, c = :blue, label = labels[2])
    if (!isnothing(axes))
        xlabel!(axes[1])
        ylabel!(axes[2])
    end
    png("images/our_plot")
end