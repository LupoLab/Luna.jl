module Hankel
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import LinearAlgebra: mul!, ldiv!, dot
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

function dimdot(v, A; dim=1)
    dims = collect(size(A))
    dims[dim] = 1
    out = Array{eltype(A)}(undef, Tuple(dims))
    dimdot!(out, v, A; dim=1)
    return out
end

function dimdot(v, A::Vector; dim=1)
    return dot(v, A)
end

function dimdot!(out, v, A; dim=1)
    idxlo = CartesianIndices(size(A)[1:dim-1])
    idxhi = CartesianIndices(size(A)[dim+1:end])
    _dimdot!(out, v, A, idxlo, idxhi)
end

function _dimdot!(out, v, A, idxlo, idxhi)
    for lo in idxlo
        for hi in idxhi
            out[lo, 1, hi] = dot(v, view(A, lo, :, hi))
        end
    end
end

mutable struct QDHT
    N::Int64 # Number of samples
    T::Array{Float64, 2} # Transform matrix
    K::Float64 # Highest spatial frequency
    k::Vector{Float64} # Spatial frequency grid
    R::Float64 # Aperture size (largest real-space coordinate)
    r::Vector{Float64} # Real-space grid
    scaleR::Vector{Float64}
    scaleK::Vector{Float64}
    dim::Int64 # Dimension along which to transform
end

function QDHT(R, N; dim=1)
    roots = besselj_zero.(0, 1:N)
    S = besselj_zero(0, N+1)
    r = roots .* R/S # real-space vector
    K = S/R # Highest spatial frequency
    k = roots .* K/S # Spatial frequency vector
    # Transform matrix
    J1 = abs.(besselj.(1, roots))
    J1sq = J1 .* J1
    T = 2/S * besselj.(0, (roots * roots')./S)./J1sq'

    scaleR = 2*(R/S)^2 ./ J1sq
    scaleK = 2*(K/S)^2 ./ J1sq
    QDHT(N, T, K, k, R, r, scaleR, scaleK, dim)
end

"Forward transform"
function mul!(Y, Q::QDHT, A)
    dot!(Y, Q.T, A, dim=Q.dim)
    Y .*= Q.R/Q.K
end

"Inverse transform"
function ldiv!(Y, Q::QDHT, A)
    dot!(Y, Q.T, A, dim=Q.dim)
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

"Radial integral"
function integrateR(A, Q)
    return dimdot(Q.scaleR, A)
end

function integrateK(A, Q)
    return dimdot(Q.scaleK, A)
end

end