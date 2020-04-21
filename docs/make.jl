using Documenter
using Luna

github_repository = "luna"
github_event_name = get(ENV, "GITHUB_EVENT_NAME", "") 
github_ref        = get(ENV, "GITHUB_REF",        "") 

cfg = Documenter.GitHubActions(github_repository, github_event_name, github_ref)

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
    repo = "lupo@luna.lupo-lab.com:/home/jtravs/webapps/luna",
    branch = "master",
    deploy_config = cfg
)