module PhysData

import PhysicalConstants: CODATA2014
import Unitful: ustrip
import Luna: Maths

"Speed of light"
const c = ustrip(CODATA2014.SpeedOfLightInVacuum)
"Pressure in Pascal at standard conditions (atmospheric pressure)"
const atm = ustrip(CODATA2014.atm)
"Boltzmann constant"
const k_B = ustrip(CODATA2014.k_B)
"Permittivity of vacuum"
const ε_0 = ustrip(CODATA2014.ε_0)
"Permeability of vacuum"
const μ_0 = ustrip(CODATA2014.μ_0)
"Electron charge"
const electron = ustrip(CODATA2014.e)
"Electron mass"
const m_e = ustrip(CODATA2014.m_e)
"Ratio of electron charge squared to electron mass (for plasma)"
const e_ratio = electron^2/m_e
"Reduced Planck's constant"
const ħ = ustrip(CODATA2014.ħ)
"Atomic unit of energy"
const au_energy = ħ*c*ustrip(CODATA2014.α)/ustrip(CODATA2014.a_0)
"Atomic unit of time"
const au_time = ħ/au_energy
"Atomic unit of electric field"
const au_Efield = au_energy/(electron*ustrip(CODATA2014.a_0))
"Room temperature in Kelvin (ca 21 deg C)"
const roomtemp = 294
"Density of an ideal gas at atmospheric pressure and room temperature"
const std_dens = atm / (k_B * roomtemp) # Gas density at standard conditions

const gas = (:Air, :He, :HeJ, :Ne, :Ar, :Kr, :Xe)
const glass = (:SiO2, :BK7, :KBr, :CaF2, :BaF2, :Si)

"Linear coefficients"

"Sellmeier expansion for linear susceptibility from Applied Optics 47, 27, 4856 (2008) at
room temperature and atmospheric pressure"
function χ_Börzsönyi(μm, B1, C1, B2, C2)
    if any(μm .> 1e4)
        throw(DomainError(μm, "Wavelength must be given in metres"))
    end
    return @. 273/roomtemp*(B1 * μm^2 / (μm^2 - C1) + B2 * μm^2 / (μm^2 - C2))
end

function χ_JCT(μm, B1, C1, B2, C2, B3, C3)
    if any(μm .> 1e4)
        throw(DomainError(μm, "Wavelength must be given in metres"))
    end
    return @. 273/roomtemp*(B1 * μm^2 / (μm^2 - C1)
                           + B2 * μm^2 / (μm^2 - C2)
                           + B3 * μm^2 / (μm^2 - C3))
end

"Sellemier expansion for gases. Return function for linear susceptibility χ which takes wavelength in
SI units and coefficients as arguments"
function sellmeier_gas(material::Symbol)
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
        throw(DomainError(material, "Unknown gas $material"))
    end
end

"Sellmeier for glasses. Returns function of wavelength in μm which in turn
returns the refractive index directly"
function sellmeier_glass(material::Symbol)
    if material == :SiO2
        #  J. Opt. Soc. Am. 55, 1205-1208 (1965)
        #TODO: Deal with sqrt of negative values better (somehow...)
        return μm -> @. sqrt(1
             + 0.6961663/(1-(0.0684043/μm)^2)
             + 0.4079426/(1-(0.1162414/μm)^2)
             + 0.8974794/(1-(9.896161/μm)^2)
             )
    elseif material == :BK7
        # ref index info (SCHOTT catalogue)
        return μm -> @. sqrt(1
             + 1.03961212/(1-0.00600069867/μm^2)
             + 0.231792344 / (1-0.0200179144/μm^2)
             + 1.01046945/(1-103.560653/μm^2)
             )
    elseif material == :CaF2
        # Appl. Opt. 41, 5275-5281 (2002)
        return μm -> @. sqrt(1
             + 0.443749998/(1-0.00178027854/μm^2)
             + 0.444930066/(1-0.00788536061/μm^2)
             + 0.150133991/(1-0.0124119491/μm^2)
             + 8.85319946/(1-2752.28175/μm^2)
             )
    elseif material == :KBr
        # J. Phys. Chem. Ref. Data 5, 329-528 (1976)
        return μm -> @. sqrt(1
             + 0.39408
             + 0.79221/(1-(0.146/μm)^2)
             + 0.01981/(1-(0.173/μm)^2)
             + 0.15587/(1-(0.187/μm)^2)
             + 0.17673/(1-(60.61/μm)^2)
             + 2.06217/(1-(87.72/μm)^2)
             )
    elseif material == :BaF2
        # J. Phys. Chem. Ref. Data 9, 161-289 (1980)
        return μm -> @. sqrt(1
             + 0.33973
             + 0.81070/(1-(0.10065/μm)^2)
             + 0.19652/(1-(29.87/μm)^2)
             + 4.52469/(1-(53.82/μm)^2)
             )
    elseif material == :Si
        # J. Opt. Soc. Am., 47, 244-246 (1957)
        return μm -> @. sqrt(1
             + 10.6684293/(1-(0.301516485/μm)^2)
             + 0.0030434748/(1-(1.13475115/μm)^2)
             + 1.54133408/(1-(1104/μm)^2)
             )
    else
        throw(DomainError(material, "Unknown glass $material"))
    end
end

"Get function to return χ1 as a function of wavelength in SI units.
Gases only."
function χ1_fun(material::Symbol)
    χ, sell = sellmeier_gas(material)
    f = let χ=χ, sell=sell
        λ -> χ(λ.*1e6, sell...)
    end
    return f
end

"Get χ1 at wavelength λ in SI units. Gases only."
function χ1(material::Symbol, λ)
    return χ1_fun(material)(λ)
end

"Helper function to get refractive index from sellmeier_gas"
function ref_index(χ::Function, sellmeier, λ,
                    pressure=1, temp=roomtemp)
    return @. sqrt(1 + roomtemp/temp * pressure * χ(λ*1e6, sellmeier...))
end

"Get refractive index for any material at wavelength given in SI units"
function ref_index(material::Symbol, λ, pressure=1, temp=roomtemp)
    return ref_index_fun(material, pressure, temp)(λ)
end

"Get function which returns refractive index."
function ref_index_fun(material::Symbol, pressure=1, temp=roomtemp)::Function
    if material in gas
        χ, sell = sellmeier_gas(material)
        ngas = let χ=χ, sell=sell, pressure=pressure, temp=temp
            ngas(λ) = ref_index(χ, sell, λ, pressure, temp)
        end
        return ngas
    elseif material in glass
        nglass = let sell = sellmeier_glass(material)
            function nglass(λ)
                if any(λ .> 1e-3)
                    throw(DomainError(λ, "Wavelength must be given in metres"))
                end
                return sell(λ.*1e6)
            end
        end
        return nglass
    else
        throw(DomainError(material, "Unknown material $material"))
    end
end

"Get a function which gives dispersion."
function dispersion_func(order, material::Symbol, pressure=1, temp=roomtemp)
    n = ref_index_fun(material, pressure, temp)
    β(ω) = @. ω/c * n(2π*c/ω)
    βn(λ) = Maths.derivative(β, 2π*c/λ, order)
    return βn
end

"Get dispersion."
function dispersion(order, material::Symbol, λ, pressure=1, temp=roomtemp)
    return dispersion_func(order, material, pressure, temp).(λ)
end

"Nonlinear coefficients"

"Calculate single-molecule third-order hyperpolarisability of a gas
at given wavelength(s) and at room temperature.
If source == :Bishop:
Uses reference values to calculate γ
If source == :Lehmeier (default):
Uses scaling factors to calculate χ3 at 1 atmosphere and scales by density
to get to a single molecule i.e. the hyperpolarisability

References:
[1] Journal of Chemical Physics, AIP, 91, 3549-3551 (1989)
[2] Chemical Reviews, 94, 3-29 (1994)
[3] Optics Communications, 56(1), 67–72 (1985)
"
function γ3_gas(material::Symbol; source=:Lehmeier)
    if source == :Lehmeier
        # Table 1 in [3]
        if material in (:He, :HeJ)
            fac = 1
        elseif material == :Ne
            fac = 1.8
        elseif material == :Ar
            fac = 23.5
        elseif material == :Kr
            fac = 64.0
        elseif material == :Xe
            fac = 188.2
        end
        return 4*fac*3.43e-28 / std_dens
    else
        error("TODO: Bishop/Shelton values for γ3")
    end
end

function χ3_gas(material::Symbol, pressure; source=:Lehmeier)
    return γ3_gas(material, source=source) .* std_dens .* pressure
end

function n2_gas(material::Symbol, pressure, λ=800e-9; source=:Lehmeier)
    n0 = ref_index(material, λ, pressure)
    return @. 3/4 * χ3_gas(material, pressure, source=source) / (ε_0*c*n0^2)
end

function ionisation_potential(material; unit=:SI)
    if material in (:He, :HeJ)
        Ip = 0.9036
    elseif material == :Ne
        Ip = 0.7925
    elseif material == :Ar
        Ip = 0.5792
    elseif material == :Kr
        Ip = 0.5142
    elseif material == :Xe
        Ip = 0.4458
    elseif material == :H
        Ip = 0.5
    else
        throw(DomainError(material, "Unknown material $material"))
    end

    if unit == :atomic
        return Ip
    elseif unit == :eV
        return @. 27.21138602*Ip
    elseif unit == :SI
        return @. 27.21138602*electron*Ip
    else
        throw(DomainError(unit, "Unknown unit $unit"))
    end
end

function quantum_numbers(material)
    # Returns n, l, ion Z
    if material == :Ar
        return 3, 1, 1
    elseif material == :Kr
        return 4, 1, 1
    elseif material in (:He, :HeJ)
        return 1, 1, 1
    end
end
end