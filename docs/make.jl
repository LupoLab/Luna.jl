using Documenter
using Luna

modulesdir = joinpath(Luna.Utils.lunadir(), "docs", "src", "modules")

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