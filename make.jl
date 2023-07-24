using Plots, Literate
using Downloads
using DelimitedFiles

ENV["GKSwstype"] = "nul"

const OUTPUT_DIR = joinpath(@__DIR__, "literated")

examples_for_literation = [
    "businger_calibration.jl"
]

for example in examples_for_literation
    example_filepath = joinpath(@__DIR__, example)
    Literate.markdown(
        example_filepath,
        OUTPUT_DIR;
        flavor = Literate.DocumenterFlavor()
    )
end