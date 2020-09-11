using Documenter
using Luna

github_repository = "luna"
github_event_name = get(ENV, "GITHUB_EVENT_NAME", "") 
github_ref        = get(ENV, "GITHUB_REF",        "") 

cfg = Documenter.GitHubActions(github_repository, github_event_name, github_ref)

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
            "$mod.jl" => "modules/$mod.md" for mod in (
                "NonlinearRHS",
                "LinearOps",
                "Modes",
                "Capillary",
                "Nonlinear",
                "PhysData",
                "Plotting",
                "Stats",
                "Ionisation",
                "Raman",
                "Output",
                "Tools"
            )
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