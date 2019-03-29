module Refraction

import PhysicalConstants
import Unitful
import Luna: Maths

const c = Unitful.ustrip(PhysicalConstants.CODATA2014.c)
const roomtemp = 294

"Sellmeier expansion for linear susceptibility from Applied Optics 47, 27, 4856 (2008) at
room temperature and atmospheric pressure"
function χ_Börzsönyi(μm, B1, C1, B2, C2)
    return @. 273/roomtemp*(B1 * μm^2 / (μm^2 - C1) + B2 * μm^2 / (μm^2 - C2))
end

function χ_JCT(μm, B1, C1, B2, C2, B3, C3)
    return @. 273/roomtemp*(B1 * μm^2 / (μm^2 - C1)
                           + B2 * μm^2 / (μm^2 - C2)
                           + B3 * μm^2 / (μm^2 - C3))
end

"Sellemier expansion. Return function for linear susceptibility χ which takes wavelength in
SI units and coefficients as arguments"
function Sellmeier(material::Symbol)
    if material == :He
        B1 = 4977.77e-8
        C1 = 28.54e-6
        B2 = 1856.94e-8
        C2 = 7.76e-3
        return χ_Börzsönyi, (B1, C1, B2, C2)
    elseif material == :HeJ
        B1 = 2.16463842e-05
        C1 = -6.80769781e-04
        B2 = 2.10561127e-07
        C2 = 5.13251289e-03
        B3 = 4.75092720e-05
        C3 = 3.18621354e-03
        return χ_JCT, (B1, C1, B2, C2, B3, C3)
    elseif material == :Ne
        B1 = 9154.48e-8
        C1 = 656.97e-6
        B2 = 4018.63e-8
        C2 = 5.728e-3
        return χ_Börzsönyi, (B1, C1, B2, C2)
    elseif material == :Ar
        B1 = 20332.29e-8
        C1 = 206.12e-6
        B2 = 34458.31e-8
        C2 = 8.066e-3
        return χ_Börzsönyi, (B1, C1, B2, C2)
    elseif material == :Kr
        B1 = 26102.88e-8
        C1 = 2.01e-6
        B2 = 56946.82e-8
        C2 = 10.043e-3
        return χ_Börzsönyi, (B1, C1, B2, C2)
    elseif material == :Xe
        B1 = 103701.61e-8
        C1 = 12750e-6
        B2 = 31228.61e-8
        C2 = 0.561e-3
        return χ_Börzsönyi, (B1, C1, B2, C2)
    elseif material == :Air
        B1 = 14926.44e-8
        C1 = 19.36e-6
        B2 = 41807.57e-8
        C2 = 7.434e-3
        return χ_Börzsönyi, (B1, C1, B2, C2)
    else
        error("Unknown material $material")
    end
end

function χ1_fun(material::Symbol)
    χ, sell = Sellmeier(material)
    f = let χ=χ, sell=sell
        λ -> χ(λ.*1e6, sell...)
    end
    return f
end

function χ1(material::Symbol, λ)
    return χ1_fun(material)(λ)
end

function ref_index(χ::Function, sellmeier, λ,
                    pressure=1, temp=roomtemp)
    return @. sqrt(1 + roomtemp/temp * pressure * χ(λ*1e6, sellmeier...))
end

function ref_index(material::Symbol, λ, pressure=1, temp=roomtemp)
    χ, sell = Sellmeier(material)
    return ref_index(χ, sell, λ, pressure, temp)
end

function ref_index_fun(material::Symbol, pressure=1, temp=roomtemp)::Function
    χ, sell = Sellmeier(material)
    n = let χ=χ, sell=sell, pressure=pressure, temp=temp
        n(λ) = ref_index(χ, sell, λ, pressure, temp)
    end
    return n
end


function dispersion_func(order, material::Symbol, pressure=1, temp=roomtemp)
    n = ref_index_fun(material, pressure, temp)
    β(ω) = @. ω/c * n(2π*c/ω)
    βn(λ) = Maths.derivative(β, 2π*c/λ, order)
    return βn
end


function dispersion(order, material::Symbol, λ, pressure=1, temp=roomtemp)
    return dispersion_func(order, material, pressure, temp).(λ)
end
end