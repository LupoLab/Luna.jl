using Documenter
using Literate
using Luna

modulesdir = joinpath(Luna.Utils.lunadir(), "docs", "src", "modules")

# Process Literate.jl source files
literate_dir = joinpath(@__DIR__, "src", "literate")
output_dir = joinpath(@__DIR__, "src", "generated")
for file in readdir(literate_dir; join=true)
    endswith(file, ".jl") || continue
    Literate.markdown(file, output_dir; documenter=true)
end

makedocs(
    sitename = "Luna Documentation",
    authors = "Christian Brahms and John C. Travers",
    pages = Any[
        "Home" => "index.md",
        "The numerical model" => [
            "General description" => "model/model.md",
            "Modal decompositions" => "model/modal_decompositions.md",
            "Nonlinear responses" => "model/nonlinear_responses.md"
        ],
        "The simple interface" => "interface.md",
        "Plotting" => "plotting.md",
        "Plotting examples" => "generated/plotting_example.md",
        "Parameter scans" => "scans.md",
        "Modules" => [
            "$(split(fi, ".")[1]).jl" => "modules/$fi" for fi in readdir(modulesdir)
        ],
    ],
    format = Documenter.HTML(
        prettyurls = false
    )
)

deploydocs(
    repo = "github.com/LupoLab/Luna.jl.git",
)
