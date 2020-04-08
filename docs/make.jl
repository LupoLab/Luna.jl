using Documenter
import Luna
import Luna: Scans

makedocs(
    sitename="Luna Documentation",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "lupo@luna.lupo-lab.com:/home/lupo/lunadocs",
    branch = "master"
)