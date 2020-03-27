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
    repo = "github.com/chrisbrahms/Luna.git",
)