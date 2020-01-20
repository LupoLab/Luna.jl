module Hankel
import FunctionZeros: besselj_zero
import SpecialFunctions: besselj
import LinearAlgebra: mul!, ldiv!, dot
import Base: *, \

J₀₀ = besselj(0, 0)

"""
    Quasi-discrete Hankel transform, after:
    [1] L. Yu, M. Huang, M. Chen, W. Chen, W. Huang, and Z. Zhu, Optics Letters 23, (1998)
    [2]M. Guizar-Sicairos and J. C. Gutiérrez-Vega, JOSA A 21, 53 (2004)
    but with some alterations (see block comment below).

    Follows AbstractFFT approach of applying fwd and inv transform with mul and ldiv
"""
mutable struct QDHT
    N::Int64 # Number of samples
    T::Array{Float64, 2} # Transform matrix
    J1sq::Array{Float64, 1} # J₁² factors
    K::Float64 # Highest spatial frequency
    k::Vector{Float64} # Spatial frequency grid
    R::Float64 # Aperture size (largest real-space coordinate)
    r::Vector{Float64} # Real-space grid
    scaleR::Vector{Float64} # Scale factor for real-space integration
    scaleK::Vector{Float64} # Scale factor for frequency-space integration
    dim::Int64 # Dimension along which to transform
end

function QDHT(R, N; dim=1)
    roots = besselj_zero.(0, 1:N)
    S = besselj_zero(0, N+1)
    r = roots .* R/S # real-space vector
    K = S/R # Highest spatial frequency
    k = roots .* K/S # Spatial frequency vector
    #= Transform matrix
    Note that this is not the same as C/T defined in [1, 2]!
    Instead of dividing by J₁(α_pn)J₁(α_pm) we divide by J₁(α_pn)^2. This cancels out
    the factor between f and F so we do not have to mutltiplydivide by J₁(α_pn/m) before and 
    after applying the transform matrix. This means T is not symmetric,
    and does not conserve energy. To conserve energy, use integrateR and integrateK (below).
    =#
    J₁ = abs.(besselj.(1, roots))
    J₁sq = J₁ .* J₁
    T = 2/S * besselj.(0, (roots * roots')./S)./J₁sq'

    scaleR = 2*(R/S)^2 ./ J₁sq
    scaleK = 2*(K/S)^2 ./ J₁sq
    QDHT(N, T, J₁sq, K, k, R, r, scaleR, scaleK, dim)
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

"Forward transform can be applied with multiplication (like FFTs)"
function *(Q::QDHT, A)
    out = similar(A)
    mul!(out, Q, A)
end

"Inverse transform can be applied with left division (like FFTs)"
function \(Q::QDHT, A)
    out = similar(A)
    ldiv!(out, Q, A)
end

"Radial integral"
function integrateR(A, Q; dim=1)
    return dimdot(Q.scaleR, A; dim=dim)
end

"Integral in conjugate space"
function integrateK(A, Q; dim=1)
    return dimdot(Q.scaleK, A; dim=dim)
end

"Compute on-axis sample from transformed array"
onaxis(A, Q) = J₀₀ .* integrateK(A, Q; dim=Q.dim)

function symmetric(A, Q)
    s = collect(size(A))
    N = s[Q.dim]
    s[Q.dim] = 2N + 1
    out = Array{eltype(A)}(undef, Tuple(s))
    idxlo = CartesianIndices(size(A)[1:Q.dim-1])
    idxhi = CartesianIndices(size(A)[Q.dim+1:end])
    out[idxlo, 1:N, idxhi] .= A[idxlo, N:-1:1, idxhi]
    out[idxlo, N+1, idxhi] .= dropdims(onaxis(Q*A, Q), dims=Q.dim)
    out[idxlo, N+2:end, idxhi] .= A[idxlo, :, idxhi]
    return out
end

Rsymmetric(Q) = vcat(-Q.r[end:-1:1], 0, Q.r)


"Matrix-vector multiplication along specific dimension of array V"
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

"Dot product between vector and one dimension of array A"
function dimdot(v, A; dim=1)
    dims = collect(size(A))
    dims[dim] = 1
    out = Array{eltype(A)}(undef, Tuple(dims))
    dimdot!(out, v, A; dim=dim)
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

end