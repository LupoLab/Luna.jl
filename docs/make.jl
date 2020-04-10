using Documenter
import Luna
import Luna: Scans

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
            "General description" => "model.md",
            "Modal decompositions" => "modal_decompositions.md",
            "Nonlinear responses" => "nonlinear_responses.md"
        ],
        "Scans" => "scans.md"
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "lupo@luna.lupo-lab.com:/home/jtravs/webapps/luna",
    branch = "master",
    deploy_config = cfg
)