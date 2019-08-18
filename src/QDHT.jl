module Hankel
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import LinearAlgebra: mul!, ldiv!
import Base: *, \

function dot!(out, M, V; dim=1)
    size(V, dim) == size(M, 1) || throw(DomainError(
        "Size of V along dim must be same as size of M"))
    idxlo = CartesianIndices(size(V)[1:dim-1])
    idxhi = CartesianIndices(size(V)[dim+1:end])
    _dot!(out, M, V, idxlo, idxhi)
end

function _dot!(out, M, V, idxlo, idxhi)
    for lo in idxlo
        for hi in idxhi
            view(out, lo, :, hi) .= M * view(V, lo, :, hi)
        end
    end
end

#Switch this to type with *, \ methods
mutable struct QDHT
    N::Int64 # Number of samples
    C::Array{Float64, 2} # Transform matrix
    K::Float64 # Highest spatial frequency
    k::Vector{Float64} # Spatial frequency grid
    R::Float64 # Aperture size (largest real-space coordinate)
    r::Vector{Float64} # Real-space grid
    dim::Int64 # Dimension along which to transform
end

function QDHT(R, N; dim=1)
    roots = besselj_zero.(0, range(1, stop=N+1))
    S = roots[end]
    roots = roots[1:end-1]
    J1sq = besselj.(1, roots).^2
    r = roots .* R/S # real-space vector
    K = S/R # Highest spatial frequency
    k = roots .* K/S # Spatial frequency vector
    # Transform matrix
    C = 2/S * besselj.(0, (roots * roots')./S)./J1sq
    QDHT(N, C, K, k, R, r, dim)
end

"Forward transform"
function mul!(Y, Q::QDHT, A)
    dot!(Y, Q.C, A, dim=Q.dim)
    Y .*= Q.R/Q.K
end

"Inverse transform"
function ldiv!(Y, Q::QDHT, A)
    dot!(Y, Q.C, A, dim=Q.dim)
    Y .*= Q.K/Q.R
end

function *(Q::QDHT, A)
    out = similar(A)
    mul!(out, Q, A)
end

function \(Q::QDHT, A)
    out = similar(A)
    ldiv!(out, Q, A)
end

end