module Polarisation

using StaticArrays
using LinearAlgebra

"Make horizontally polarised Jones vector"
function H()
   SVector(1.0, 0.0)
end

"Make horizontally orientated linear polariser Jones matrix"
function LP()
    @SMatrix [ 1.0  0.0 ;
               0.0  0.0 ]
end

"Arbitrary waveplate with phase ϕ; fast axis horizontal"
function WP(ϕ)
    @SMatrix [ exp(im*ϕ/2)  0.0 ;
               0.0          exp(-im*ϕ/2) ]
end

"rotation operator"
function rot(θ)
    @SMatrix [ cos(θ)  sin(θ) ;
              -sin(θ)  cos(θ) ]
end 

"rotate Jonesmatrix J by θ"
function rotate(J, θ)
    rot(-θ)*J*rot(θ)
end

"Get Stokes parameters for input field E = (Ex, Ey)"
function Stokes(E; normalise=false)
    Ex = E[1]
    Ey = E[2]
    I = abs(Ex)^2 + abs(Ey)^2
    Q = abs(Ex)^2 - abs(Ey)^2
    U = 2*real(Ex*conj(Ey))
    V = -2*imag(Ex*conj(Ey))
    S = SVector(I, Q, U, V)
    if normalise
        S = S./S[1]
    end
    S
end

"Get normalised cartesian coordinates from Stokes parameters
 (for Poincare sphere) "
function cartesian(S)
    S[2:end]./S[1]
end

"Get polarization ellipse parameters from Stokes parameters"
function ellipse(S)
    I = S[1]
    Q = S[2]
    U = S[3]
    V = S[4]
    aL = sqrt(Q^2 + U^2)
    θ = angle(aL)/2
    A = sqrt((I + aL)/2)
    B = sqrt((abs(I - aL))/2)
    h = sign(V)
    A, B, θ, h
end

"Calculate ellipticity from Stokes parameters"
function ellipticity(S)
    A, B, θ, h = ellipse(S)
    r = A/B
    r > 1 ? 1/r : r
end

end
