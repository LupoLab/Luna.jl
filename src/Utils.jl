module Utils
import Dates

function git_commit()
    wd = dirname(@__FILE__)
    commit = read(`git -C $wd describe --always --tags --dirty`, String)
    commit = commit[1:end-1] # Strip newline off the end
end

function git_isdirty()
    wd = dirname(@__FILE__)
    d = read(`git -C $wd diff --name-only`)
    return length(d) > 0
end

function git_branch()
    wd = dirname(@__FILE__)
    b = read(`git rev-parse --abbrev-ref HEAD`, String)
    return b[1:end-1] # Strip newline off the end
end

function sourcecode()
    src = dirname(@__FILE__)
    luna = dirname(src)
    out = "#= Date: $(Dates.now())\n"
    out *= "git branch: $(git_branch())\n"
    out *= "git commit: $(git_commit())\n"
    out *= "hostname: $(gethostname())\n"
    out *= "=#"
    for folder in (src, luna)
        for obj in readdir(folder)
            if isfile(joinpath(folder, obj))
                if split(obj, ".")[end] in ("md", "jl", "txt", "toml") # avoid binary files
                    out *= "\n" * "#" * "="^8 * obj * "="^8 * "#" * "\n"^2
                    open(joinpath(folder, obj), "r") do file
                        out *= read(file, String)
                    end
                end
            end
        end
    end
    return out
end
end