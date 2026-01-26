# from https://github.com/GeoscienceAustralia/HiQGA.jl/commit/db833bf32840503ee3bd0909b2b92993c239413c#diff-708e1c220f34be9ffbf04c0619b1f1a56388096c1df2b95603950d7adc80feaa
import Pkg, Conda
@info "building!"
# Conda.pip_interop(true)
# Conda.pip("install", "matplotlib")
Conda.add("matplotlib")
ENV["PYTHON"] = joinpath(Conda.ROOTENV, "bin", "python")
Pkg.build("PyCall")
@info "built PyCall!"
