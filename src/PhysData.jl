module PhysData

import CoolProp
import PhysicalConstants: CODATA2014
import Unitful: ustrip
import CSV
import DelimitedFiles: readdlm
import Polynomials
import Luna: Maths, Utils

include("data/lookup_tables.jl")

"Speed of light"
const c = ustrip(CODATA2014.SpeedOfLightInVacuum)
"Pressure in Pascal at standard conditions (atmospheric pressure)"
const atm = ustrip(CODATA2014.atm)
"Pressure in Pascal of 1 bar"
const bar = 100000
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
"Room temperature in Kelvin (ca 20 deg C)"
const roomtemp = 293.15
"Avogadro constant"
const N_A = ustrip(CODATA2014.N_A)
"Amagat (Loschmidt constant)"
const amg = atm/(k_B*273.15)
"Atomic mass unit"
const m_u = ustrip(CODATA2014.m_u)
"Atomic unit of electric polarisability"
const au_polarisability = electron^2*ustrip(CODATA2014.a_0)^2/au_energy

const gas = (:Air, :He, :HeJ, :HeB, :Ne, :Ar, :Kr, :Xe, :N2, :H2, :O2, :CH4, :SF6, :N2O, :D2)
const gas_str = Dict(
    :He => "He",
    :HeB => "He",
    :HeJ => "He",
    :Ar => "Ar",
    :Ne => "Neon",
    :Kr => "Krypton",
    :Xe => "Xenon",
    :Air => "Air",
    :N2 => "Nitrogen",
    :H2 => "Hydrogen",
    :O2 => "Oxygen",
    :CH4 => "Methane",
    :SF6 => "SulfurHexafluoride",
    :N2O => "NitrousOxide",
    :D2 => "Deuterium"
)
const glass = (:SiO2, :BK7, :KBr, :CaF2, :BaF2, :Si, :MgF2, :ADPo, :ADPe, :KDPo, :KDPe, :CaCO3)
const metal = (:Ag,:Al)

"""
    wlfreq(ωλ)

Change from ω (angular frequency) to λ (wavelength) and vice versa
"""
wlfreq(ωλ) = 2π*c/ωλ

"""
    ΔλΔω(Δλ, λ)

Convert Δλ (wavelength bandwidth) at λ (central wavelength) to Δω (angular frequency bandwidth)
"""
ΔλΔω(Δλ, λ) = (2π*c)*Δλ/λ^2

eV_to_m(eV) = wlfreq(electron*eV/ħ)


"""
    γ_Börzsönyi(B1, C1, B2, C2)

Sellmeier expansion for linear susceptibility from Applied Optics 47, 27, 4856 (2008) at
room temperature and atmospheric pressure
"""
function γ_Börzsönyi(B1, C1, B2, C2)
    return μm -> (B1 * μm^2 / (μm^2 - C1) + B2 * μm^2 / (μm^2 - C2))
end

"""
    γ_JCT(B1, C1, B2, C2, B3, C3)

Adapted Sellmeier expansion for helium made to fit high frequency data
Phys. Rev. A 92, 033821 (2015)
"""
function γ_JCT(B1, C1, B2, C2, B3, C3)
    return μm -> (B1 * μm^2 / (μm^2 - C1)
                  + B2 * μm^2 / (μm^2 - C2)
                  + B3 * μm^2 / (μm^2 - C3))
end

"""
    γ_Peck(B1, C1, B2, C2, dens)

Sellmeier expansion for linear susceptibility from
J. Opt. Soc. Am. 67, 1550 (1977)
"""
function γ_Peck(B1, C1, B2, C2, dens)
    return μm -> (((B1 / (C1 - 1/μm^2) + B2 / (C2 - 1/μm^2)) + 1)^2 - 1)/dens
end

"""
    γ_Zhang(A, B, C, dens)

Sellmeier expansion for Oxygen from Applied Optics 50, 35, 6484 (2011)
"""
function γ_Zhang(A, B, C, dens)
    return μm -> ((1 + A + B/(C-1/μm^2))^2 - 1)/dens
end

"""
    γ_QuanfuHe(A, B, C, dens)

Sellmeier expansion for CH4, SF6 and N2O from Atmospheric Chemistry and Physics 2021, 21 (19), 14927–14940.
https://doi.org/10.5194/acp-21-14927-2021.

"""
function γ_QuanfuHe(A, B, C, dens)
    return μm -> complex((1 + 1e-8*(A + B/(C - (1e4/μm)^2))))/dens
end

"""
    sellmeier_gas(material::Symbol)

Return function for linear polarisability γ, i.e. susceptibility of a single particle,
calculated from Sellmeier expansions.
"""
function sellmeier_gas(material::Symbol)
    dens = dens_1bar_0degC[material]
    if material == :HeB
        B1 = 4977.77e-8
        C1 = 28.54e-6
        B2 = 1856.94e-8
        C2 = 7.76e-3
        return γ_Börzsönyi(B1/dens, C1, B2/dens, C2)
    elseif material == :He || material == :HeJ
        B1 = 2.16463842e-05
        C1 = -6.80769781e-04
        B2 = 2.10561127e-07
        C2 = 5.13251289e-03
        B3 = 4.75092720e-05
        C3 = 3.18621354e-03
        return γ_JCT(B1/dens, C1, B2/dens, C2, B3/dens, C3)
    elseif material == :Ne
        B1 = 9154.48e-8
        C1 = 656.97e-6
        B2 = 4018.63e-8
        C2 = 5.728e-3
        return γ_Börzsönyi(B1/dens, C1, B2/dens, C2)
    elseif material == :Ar
        B1 = 20332.29e-8
        C1 = 206.12e-6
        B2 = 34458.31e-8
        C2 = 8.066e-3
        return γ_Börzsönyi(B1/dens, C1, B2/dens, C2)
    elseif material == :Kr
        B1 = 26102.88e-8
        C1 = 2.01e-6
        B2 = 56946.82e-8
        C2 = 10.043e-3
        return γ_Börzsönyi(B1/dens, C1, B2/dens, C2)
    elseif material == :Xe
        B1 = 103701.61e-8
        C1 = 12750e-6
        B2 = 31228.61e-8
        C2 = 0.561e-3
        return γ_Börzsönyi(B1/dens, C1, B2/dens, C2)
    elseif material == :Air
        B1 = 14926.44e-8
        C1 = 19.36e-6
        B2 = 41807.57e-8
        C2 = 7.434e-3
        return γ_Börzsönyi(B1/dens, C1, B2/dens, C2)
    elseif material == :N2
        B1 = 39209.95e-8
        C1 = 1146.24e-6
        B2 = 18806.48e-8
        C2 = 13.476e-3
        return γ_Börzsönyi(B1/dens, C1, B2/dens, C2)
    elseif material in (:H2, :D2)
        # for D2 it is essentially the same as H2 according to:
        # Orr, W. J. C. "The refractive index of deuterium."
        # Transactions of the Faraday Society 32 (1936): 1556-1559.
        B1 = 14895.6e-6
        C1 = 180.7
        B2 = 4903.7e-6
        C2 = 92.0
        return γ_Peck(B1, C1, B2, C2, density(material, atm/bar, 273.15))
    elseif material == :O2
        # Applied Optics 50, 35, 6484 (2011)
        A = 1.181494e-4
        B = 9.708931e-3
        C = 75.4
        return γ_Zhang(A, B, C, density(material, atm/bar, roomtemp))
    elseif material == :CH4
        # Atmospheric Chemistry and Physics 2021, 21 (19), 14927–14940.
        A = 3603.09
        B = 4.40362e14
        C = 1.1741e10
        return  γ_QuanfuHe(A, B, C, density(material, atm/bar, 288.15))
    elseif material == :N2O
        # Atmospheric Chemistry and Physics 2021, 21 (19), 14927–14940.
        A = 22095.0
        B = 1.66291e14
        C = 6.75226e9
        return  γ_QuanfuHe(A, B, C, density(material, atm/bar, 288.15))
    elseif material == :SF6
        # Atmospheric Chemistry and Physics 2021, 21 (19), 14927–14940.
        A = 18997.7
        B = 8.27663e14
        C = 1.56833e10
        return  γ_QuanfuHe(A, B, C, density(material, atm/bar, 288.15))
    else
        throw(DomainError(material, "Unknown gas $material"))
    end
end

"""
    sellmeier_glass(material::Symbol)

Sellmeier for glasses. Returns function of wavelength in μm which in turn returns the
refractive index directly
"""
function sellmeier_glass(material::Symbol)
    if material == :SiO2
        #  J. Opt. Soc. Am. 55, 1205-1208 (1965)
        # TODO: Deal with sqrt of negative values better (somehow...)
        return μm -> sqrt(complex(1
             + 0.6961663/(1-(0.0684043/μm)^2)
             + 0.4079426/(1-(0.1162414/μm)^2)
             + 0.8974794/(1-(9.896161/μm)^2)
             ))
    elseif material == :BK7
        # ref index info (SCHOTT catalogue)
        return μm -> sqrt(complex(1
             + 1.03961212/(1-0.00600069867/μm^2)
             + 0.231792344 / (1-0.0200179144/μm^2)
             + 1.01046945/(1-103.560653/μm^2)
             ))
    elseif material == :CaF2
        # Appl. Opt. 41, 5275-5281 (2002)
        return μm -> sqrt(complex(1
             + 0.443749998/(1-0.00178027854/μm^2)
             + 0.444930066/(1-0.00788536061/μm^2)
             + 0.150133991/(1-0.0124119491/μm^2)
             + 8.85319946/(1-2752.28175/μm^2)
             ))
    elseif material == :KBr
        # J. Phys. Chem. Ref. Data 5, 329-528 (1976)
        return μm -> sqrt(complex(1
             + 0.39408
             + 0.79221/(1-(0.146/μm)^2)
             + 0.01981/(1-(0.173/μm)^2)
             + 0.15587/(1-(0.187/μm)^2)
             + 0.17673/(1-(60.61/μm)^2)
             + 2.06217/(1-(87.72/μm)^2)
             ))
    elseif material == :BaF2
        # J. Phys. Chem. Ref. Data 9, 161-289 (1980)
        return μm -> sqrt(complex(1
             + 0.33973
             + 0.81070/(1-(0.10065/μm)^2)
             + 0.19652/(1-(29.87/μm)^2)
             + 4.52469/(1-(53.82/μm)^2)
             ))
    elseif material == :Si
        # J. Opt. Soc. Am., 47, 244-246 (1957)
        return μm -> sqrt(complex(1
             + 10.6684293/(1-(0.301516485/μm)^2)
             + 0.0030434748/(1-(1.13475115/μm)^2)
             + 1.54133408/(1-(1104/μm)^2)
             ))
    elseif material == :MgF2
        return μm -> @. sqrt(complex(1
            + 0.27620
            + 0.60967/(1-(0.08636/μm)^2)
            + 0.0080/(1-(18.0/μm)^2)
            + 2.14973/(1-(25.0/μm)^2)
            ))
    elseif material == :CaCO3
        return μm -> @. sqrt(complex(1
            + 0.73358749
            + 0.96464345/(1-1.94325203e-2/μm^2)
            + 1.82831454/(1-120/μm^2)
            ))
    elseif material == :ADPo
        return μm -> @. sqrt(complex(
            2.302842
            + 15.102464*μm^2/(μm^2-400)
            + 0.011125165/(μm^2-0.01325366)
        ))
    elseif material == :ADPe
        return μm -> @. sqrt(complex(
            2.163510
            + 5.919896*μm^2/(μm^2-400)
            + 0.009616676/(μm^2-0.01298912)
        ))
    elseif material == :KDPo
        return μm -> @. sqrt(complex(
            2.259276
            + 13.00522*μm^2/(μm^2-400)
            + 0.01008956/(μm^2-0.0129426)
        ))
    elseif material == :KDPe
        return μm -> @. sqrt(complex(
            2.132668
            + 3.2279924*μm^2/(μm^2-400)
            + 0.008637494/(μm^2-0.0122810)
        ))
    else
        throw(DomainError(material, "Unknown glass $material"))
    end
end

"""
    sellmeier_crystal(material, axis)

Sellmeier for crystals. Returns function of wavelength in μm which in turn returns the
refractive index directly. Possible values for `axis` depend on the type of crystal.
"""
function sellmeier_crystal(material, axis)
    if material == :BBO
        if axis == :o
            return μm -> sqrt(complex(1
                + 0.90291/(1-0.003926/μm^2)
                + 0.83155/(1-0.018786/μm^2)
                + 0.76536/(1-60.01/μm^2)
                ))
        elseif axis == :e
            return μm -> sqrt(complex(1
                + 1.151075/(1-0.007142/μm^2)
                + 0.21803/(1-0.02259/μm^2)
                + 0.656/(1-263/μm^2)
                ))
        else
            throw(DomainError(axis, "Unknown BBO axis $axis"))
        end
    elseif material == :LBO
        # C Chen et al., J Opt. Soc. Am. 6, 616-621 (1989)
        # F. Hanson and D. Dick., Opt. Lett. 16, 205-207 (1991).
        if axis == :x
            return μm -> sqrt(complex(
                2.45768
                + 0.0098877/(μm^2-0.026095)
                - 0.013847*μm^2
            ))
        elseif axis == :y
            return μm -> sqrt(complex(
                2.52500
                + 0.017123/(μm^2+0.0060517)
                - 0.0087838*μm^2
            ))
        elseif axis == :z
            return μm -> sqrt(complex(
                2.58488
                + 0.012737/(μm^2-0.021414)
                - 0.016293*μm^2
            ))
        else
            throw(DomainError(axis, "Unknown LBO axis $axis"))
        end
    else
        throw(DomainError(material, "Unknown crystal $material"))
    end
end

function ref_index_fun_uniax(material; axes=(:o, :e))
    n_o = sellmeier_crystal(material, axes[1])
    n_e = sellmeier_crystal(material, axes[2])
    n(λ, θ) = sqrt(1/((cos(θ)/n_o(λ*1e6))^2+(sin(θ)/n_e(λ*1e6))^2))
    return n
end

"""
    χ1_fun(gas::Symbol)

Get function to return χ1 (linear susceptibility) for gases as a function of
wavelength in SI units, pressure in bar, and temperature in Kelvin.
"""
function χ1_fun(gas::Symbol)
    γ = sellmeier_gas(gas)
    f = let γ=γ, gas=gas
        function χ1(λ, P, T)
            γ(λ*1e6)*density(gas, P, T)
        end
    end
    return f
end

function χ1_fun(gas::Symbol, P, T)
    γ = sellmeier_gas(gas)
    dens = density(gas, P, T)
    return λ -> γ(λ*1e6)*dens
end

"""
    χ1(gas::Symbol, λ, P=1.0, T=roomtemp)

Calculate χ1 at wavelength λ in SI units, pressure P in bar and temperature T in Kelvin.
Gases only.
"""
function χ1(gas::Symbol, λ, P=1.0, T=roomtemp)
    return χ1_fun(gas)(λ, P, T)
end


"""
    ref_index(material, λ, P=1.0, T=roomtemp; lookup=nothing)

Get refractive index for any material at wavelength given in SI units.
"""
function ref_index(material, λ, P=1.0, T=roomtemp; lookup=nothing)
    return ref_index_fun(material, P, T; lookup=lookup)(λ)
end

"""
    ref_index_fun(material::Symbol, P=1.0, T=roomtemp; lookup=nothing)

Get function which returns refractive index.
"""
function ref_index_fun(material::Symbol, P=1.0, T=roomtemp; lookup=nothing)
    if material in gas
        χ1 = χ1_fun(material, P, T)
        return λ -> sqrt(1 + complex(χ1(λ)))
    elseif material in glass
        if isnothing(lookup)
            lookup = (material == :SiO2)
        end
        if lookup
            spl = lookup_glass(material)
            return λ -> spl(λ*1e6)
        else
            sell = sellmeier_glass(material)
            return λ -> sell(λ*1e6)
        end
    elseif material in metal
        nmetal = let spl = lookup_metal(material)
            function nmetal(λ)
                if λ > 1
                    throw(DomainError(λ, "Wavelength must be given in metres"))
                end
                return spl(λ*1e6)
            end
        end
        return nmetal
    else
        throw(DomainError(material, "Unknown material $material"))
    end
end

"""
    ref_index_fun(gases, P, T=roomtemp; lookup=nothing)

Get function which returns refractive index for gas mixture. `gases` is a `Tuple` of gas
identifiers (`Symbol`s) and `P` is a `Tuple` of equal length containing pressures.
"""
function ref_index_fun(gases::NTuple{N, Symbol}, P::NTuple{N, Number}, T=roomtemp; lookup=nothing) where N
    ngas = let funs=[χ1_fun(gi, Pi, T) for (gi, Pi) in zip(gases, P)]
        function ngas(λ)
            res = funs[1](λ)
            for ii in 2:length(gases) 
                res += funs[ii](λ)
            end
            return sqrt(1 + res)
        end
    end
    return ngas
end

"""
    ref_index_fun(gases, T=roomtemp)

Get function which returns ref index for mixture as function of wavelength and densities.
"""
function ref_index_fun(gases::NTuple{N, Symbol}, T=roomtemp) where N
    let γs=[sellmeier_gas(gi) for gi in gases]
        function ngas(λ, densities::Vector{<:Number})
            χ1 = 0.0
            for (γi, di) in zip(γs, densities)
                χ1 += di * γi(λ*1e6)
            end
            sqrt(1 + χ1)
        end
    end
end


"""
    dispersion_func(order, n)

Get a function that calculates dispersion of order `order` for a refractive index given by
`n(λ)`.
"""
function dispersion_func(order, n)
    β(ω) = @. ω/c * real(n(wlfreq(ω)))
    βn(λ) = Maths.derivative(β, wlfreq(λ), order)
    return βn
end

"""
    dispersion_func(order, material, P=1, T=roomtemp; lookup=nothing)

Get a function to calculate dispersion. Arguments are the same as for [`dispersion`](@ref).
"""
function dispersion_func(order, material::Symbol, P=1.0, T=roomtemp; lookup=nothing)
    n = ref_index_fun(material, P, T, lookup=lookup)
    dispersion_func(order, n)
end

"""
    dispersion(order, material, λ, P=1.0, T=roomtemp; lookup=nothing)

Calculate the dispersion of order `order` of a given `material` at a wavelength `λ`.

For gases the pressure `P` (default:atmosphere) and the temperature `T` (default: room temp)
can also be specified. `lookup::Bool` determines whether a lookup table or a Sellmeier 
expansion is used for the refractive index (default is material dependent).

# Examples
```jldoctest
julia> dispersion(2, :BK7, 400e-9) * 1e30 * 1e-3 # convert to fs^2/mm
122.03632107303108
```
"""
function dispersion(order, material::Symbol, λ, P=1.0, T=roomtemp; lookup=nothing)
    return dispersion_func(order, material, P, T; lookup=lookup).(λ)
end

"""
    fresnel(n2, θi; n1=1.0)

Calcualte reflection coefficients from Fresnel's equations.
"""
function fresnel(n2, θi; n1=1.0)
    θt = asin(n1*sin(θi)/n2)
    rs = (n1*cos(θi) - n2*cos(θt))/(n1*cos(θi) + n2*cos(θt))
    rp = (n2*cos(θi) - n1*cos(θt))/(n2*cos(θi) + n1*cos(θt))
    abs2(rs), angle(rs), abs2(rp), angle(rp)
end


"""
    γ3_gas(material::Symbol; source=nothing)

Calculate single-molecule third-order hyperpolarisability of a gas at given wavelength(s)
and at room temperature.
If `source` == `:Bishop`: Uses reference values to calculate γ
If `source` == `:Lehmeier` (default): Uses scaling factors to calculate χ3 at 1 bar and
scales by density to get to a single molecule i.e. the hyperpolarisability

References:
[1] Journal of Chemical Physics, AIP, 91, 3549-3551 (1989)
[2] Chemical Reviews, 94, 3-29 (1994)
[3] Optics Communications, 56(1), 67–72 (1985)
[4] Phys. Rev. A, vol. 42, 2578 (1990)
[5] Optics Letters Vol. 40, No. 24 (2015))
[6] Phys. Rev. A 2012, 85 (4), 043820. https://doi.org/10.1103/PhysRevA.85.043820.
[7] Phys. Rev. A, 32, no. 6, 3454, (1985), doi: 10.1103/PhysRevA.32.3454.

"""
function γ3_gas(material::Symbol; source=nothing)
    # TODO: More Bishop/Shelton; Wahlstrand updated values.
    if source === nothing
        if material in (:He, :HeB, :HeJ, :Ne, :Ar, :Kr, :Xe, :N2)
            source = :Lehmeier
        elseif material in (:H2, :CH4, :SF6, :D2)
            source = :Shelton
        elseif material in (:O2,)
            source = :Zahedpour
        elseif material in (:N2O,)
            source = :Wahlstrand
        else
            error("no default γ3 source for material: $material")
        end
    end
    if source == :Lehmeier
        dens = dens_1atm_0degC[material]
        # Table 1 in [3]
        if material in (:He, :HeB, :HeJ)
            fac = 1
        elseif material == :Ne
            fac = 1.8
        elseif material == :Ar
            fac = 23.5
        elseif material == :Kr
            fac = 64.0
        elseif material == :Xe
            fac = 188.2
        elseif material == :N2
            fac = 21.1
        else
            throw(DomainError(material, "Lehmeier model does not include $material"))
        end
        return 4*fac*3.43e-28 / dens
    elseif source == :Shelton
        # ref [4], we use Table 1 to simply scale from
        # the paired gas (for which we use Lehmeier)
        # e.g. He for H2, N2 for CH4 or SF6
        # for D2 we know from [7] that it is basically the same as :H2.
        if material in (:H2, :D2)
            return 15.77*γ3_gas(:He)
        elseif material == :CH4
            return 2.931*γ3_gas(:N2)
        elseif material == :SF6
            return 1.53*γ3_gas(:N2)
        else
            throw(DomainError(material, "Shelton model does not include $material"))
        end
    elseif source == :Zahedpour
        if material == :O2
            n0 = ref_index(:O2, 800e-9, atm/bar, roomtemp)
            ρ = density(:O2, atm/bar, roomtemp)
            n2 = 8.1e-24 # Table 1 in [5]
            return 4/3*ε_0*c*n0^2/ρ * n2
        else
            throw(DomainError(material, "Zahedpour model does not include $material"))
        end
    elseif source == :Wahlstrand
        if material == :N2O
            n0 = ref_index(:N2O, 800e-9, atm/bar, roomtemp)
            ρ = density(:N2O, atm/bar, roomtemp)
            n2 = 17.2e-24 # Table 1 in [6]
            return 4/3*ε_0*c*n0^2/ρ * n2
        else
            throw(DomainError(material, "Wahlstrand model does not include $material"))
        end
    else
        throw(DomainError(source, "Unkown γ3 model $source"))
    end
end

function χ3(material::Symbol, P=1.0, T=roomtemp; source=nothing)
    if material in glass
        n2 = n2_glass(material, λ=1030e-9)
        n0 = real(ref_index(material, 1030e-9))
        return 4/3 * n2 * (ε_0*c*n0^2)
    end
    return γ3_gas(material, source=source) .* density.(material, P, T)
end

function n2(material::Symbol, P=1.0, T=roomtemp; λ=nothing, source=nothing)
    material in glass && return n2_glass(material::Symbol, λ=λ)
    λ = isnothing(λ) ? 800e-9 : λ
    n0 = ref_index(material, λ, P, T)
    return @. 3/4 * χ3(material, P, T, source=source) / (ε_0*c*n0^2)
end

function n2_glass(material::Symbol; λ=nothing)
    if material == :SiO2
        return 2.7e-20
    elseif material == :MgF2
        # R. DeSalvo et al., IEEE J. Q. Elec. 32, 10 (1996).
        return 5.79e-21
    elseif material == :CaCO3
        # Kabaciński et al., 10.1364/OE.27.011018
        return 3.22e-20
    else
        error("Unkown glass $material")
    end
end

"""
    density(material::Symbol, P=1.0, T=roomtemp)

For a gas `material`, return the number density [m^-3] at pressure `P` [bar] and temperature `T` [K].
For a glass, this simply returns 1.0.
"""
function density(material::Symbol, P=1.0, T=roomtemp)
    material in glass && return 1.0
    P == 0 ? zero(P) : CoolProp.PropsSI("DMOLAR", "T", T, "P", bar*P, gas_str[material])*N_A
end

dens_1bar_0degC = Dict(gi => density(gi, 1.0, 273.15) for gi in gas)
dens_1atm_0degC = Dict(gi => density(gi, atm/bar, 273.15) for gi in gas)

"""
    pressure(gas, density, T=roomtemp)

Calculate the pressure in bar of the `gas` at number density `density` and temperature `T`.
"""
function pressure(gas, density, T=roomtemp)
    density == 0 ? zero(density) :
                   CoolProp.PropsSI("P", "T", T, "DMOLAR", density/N_A, gas_str[gas])/bar
end

"""
    densityspline(gas; Pmax, Pmin=0, N=2^10, T=roomtemp)

Create a `CSpline` interpolant for the density of the `gas` between pressures `Pmin` and
`Pmax` at temperature `T`. The spline is created using `N` samples.
"""
function densityspline(gas::Symbol; Pmax, Pmin=0, N=2^10, T=roomtemp)
    P = collect(range(Pmin, Pmax, length=N))
    ρ = density.(gas, P, T)
    Maths.CSpline(P, ρ)
end

"""
    ionisation_potential(material; unit=:SI)

Return the first ionisation potential of the `material` in a specific unit (default: SI).
Possible units are `:SI`, `:atomic` and `:eV`.
"""
function ionisation_potential(material; unit=:SI)
    if material in (:He, :HeB, :HeJ)
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
    elseif material == :N2
        Ip = 0.5726
    elseif material == :H2
        Ip = 0.5669
    elseif material == :O2
        Ip = 0.443553
    elseif material == :CH4
        Ip = 0.4636
    elseif material == :N2O
        Ip = 0.474
    elseif material == :SF6
        Ip = 0.5
    elseif material == :D2
        Ip = 0.5684 # from NIST Chemistry WebBook
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

"""
    quantum_numbers(material)

Return the quantum numbers of the `material` for use in the PPT ionisation rate.
"""
function quantum_numbers(material)
    # Returns n, l, ion Z
    if material == :Ar
        return 3, 1, 1
    elseif material == :Ne
        return 2, 1, 1;
    elseif material == :Kr
        return 4, 1, 1
    elseif material == :Xe
        return 5, 1, 1
    elseif material in (:He, :HeB, :HeJ)
        return 1, 0, 1
    elseif material == :O2
        return 2, 0, 0.53 # https://doi.org/10.1016/S0030-4018(99)00113-3
    elseif material == :N2
        return 2, 0, 0.9 # https://doi.org/10.1016/S0030-4018(99)00113-3
    else
        throw(DomainError(material, "Unknown material $material"))
    end
end

"""
    polarisability(material, ion=false; unit=:SI)

Return the polarisability of the ground state or the ion for the
`material`. `unit` can be `:SI` or `:atomic`

Data exists for helium, neon and argon. For other `material`s,
return `missing`.

Reference:
Wang, K. et al.
Static dipole polarizabilities of atoms and ions from Z=1 to 20
calculated within a single theoretical scheme.
Eur. Phys. J. D 75, 46 (2021).

"""
function polarisability(material, ion=false; unit=:SI)
    if unit == :SI
        factor = au_polarisability
    elseif unit == :atomic
        factor = 1
    else
        throw(DomainError(unit, "Unknown unit $unit"))
    end
    if material in (:He, :HeB, :HeJ)
        return (ion ? 0.2811 : 1.3207)*factor
    elseif material == :Ne
        return (ion ? 1.2417 : 2.376)*factor
    elseif material == :Ar
        return (ion ? 6.807 : 10.762)*factor
    else
        return missing
    end
end


"""
    polarisability_difference(material; unit=:SI)

Return the difference in polarisability between the ground state and the ion for the
`material`. `unit` can be `:SI` or `:atomic`

Reference:
Wang, K. et al.
Static dipole polarizabilities of atoms and ions from Z=1 to 20
calculated within a single theoretical scheme.
Eur. Phys. J. D 75, 46 (2021).

"""
function polarisability_difference(material; unit=:SI)
    polarisability(material, false; unit) - polarisability(material, true; unit)
end

"""
    Cnl_ADK(material)

Return the value of Cₙₗ from the ADK paper for the `material`. For `material`S
other than noble gases, this returns `missing`.

Reference:
Ammosov, M. V., Delone, N. B. & Krainov, V. P. Tunnel Ionization Of Complex Atoms And Atomic Ions In Electromagnetic Field. Soviet Physics JETP 64, 1191–1194 (1986).
"""
function Cnl_ADK(material)
    if material in (:He, :HeB, :HeJ)
        return 1.99
    elseif material == :Ne
        return 1.31
    elseif material == :Ar
        return 1.9
    elseif material == :Kr
        return 2.17
    elseif material == :Xe
        return 2.27
    else
        return missing
    end
end

"""
    lookup_glass(material::Symbol)

Create a `CSpline` interpolant for look-up-table values of the refractive index.
"""
function lookup_glass(material::Symbol)
    if material == :SiO2
        data = data_glass(:SiO2)
        spl = Maths.BSpline(1e6*eV_to_m.(data[:, 1]), data[:, 2] + 1im * data[:, 3])
    else
        throw(DomainError(material, "Unknown metal $material"))
    end
    return spl
end

"""
    lookup_metal(material::Symbol)

Create a `CSpline` interpolant for look-up-table values of the refractive index.
"""
function lookup_metal(material::Symbol)
    data = data_metal(material)::Array{Float64,2}
    nspl = Maths.BSpline(data[:,1], data[:,2] .+ im.*data[:,3])
    return nspl
end

"""
    raman_parameters(material)

Get the Raman parameters for `material`.

# Fields
Fields in the returned named tuple must include:
- `kind::Symbol`: one of `:molecular` or `:intermediate` or `:normedsdo`

If `kind == :molecular` then the following must also be specified:
- `rotation::Symbol`: only `:nonrigid` or `:none` supported at present.
- `vibration::Symbol`: only `:sdo` or `:none` supported at present.

If `rotation == :nonrigid` then the following must also be specified:
- `B::Real`: the rotational constant [1/m]
- `Δα::Real`: molecular polarizability anisotropy [m^3]
- `qJodd::Integer`: nuclear spin parameter for odd `J`
- `qJeven::Integer`: nuclear spin parameter for even `J`
- `D::Real=0.0`: centrifugal constant [1/m]
Along with one of:
- `τ2r::Real`: coherence time [s]
- `Bρr::Real` : density dependent broadening coefficient [Hz/amagat]
If both `τ2r` and `Bρr` are specified, then `Bρr` takes precedence.
If `Bρr` is specified then we also need:
- `Aρr::Real` : self diffusion coefficient [Hz amagat]

If `vibration == :sdo` then the following must also be specified:
- `Ωv::Real`: vibrational frequency [rad/s]
- `dαdQ::Real`: isotropic averaged polarizability derivative [m^2]
- `μ::Real`: reduced molecular mass [kg]
Along with one of:
- `τ2v::Real`: coherence time [s]
- `Bρv::Real` : density dependent broadening coefficient [Hz/amagat]
If both `τ2v` and `Bρv` are specified, then `Bρv` takes precedence.
If `Bρv` is specified then we also need:
- `Aρv::Real` : self diffusion coefficient [Hz amagat]
And can also add (if necessary) a constant offset:
- `Cv::Real` : constant linewidth offset [Hz]

If `kind == :intermediate` then the following must be specified
- `ωi::Vector{Real}` [rad/s], central angular freqencies
- `Ai::Vector{Real}`, amplitudes
- `Γi::Vector{Real}` [rad/s], Gaussian widths
- `γi::Vector{Real}` [rad/s], Lorentzian widths

# References
[1] Phys. Rev. A, 94, 023816 (2016)
[2] Phys. Rev. A, 85, 043820 (2012)
[3] Phys. Rev. A, 92, 063828 (2015)
[4] Journal of Raman Spectroscopy 2, 133 (1974)
[5] J. Phys. Chem., 91, 41 (1987)
[6] Applied Spectroscopy 23, 211 (1969)
[7] Phys. Rev. A, 34, 3, 1944 (1986)
[8] Can. J. Phys., 44, 4, 797 (1966)
[9] G. V. MIKHAtLOV, SOVIET PHYSICS JETP, vol. 36, no. 9, (1959).
[10] Phys. Rev. A, 33, 5, 3113 (1986)
[11] IEEE Journal of Quantum Electronics 1986, 22 (2), 332–336. https://doi.org/10.1109/JQE.1986.1072945.
[12] Phys. Rev. Lett. 1998, 81 (6), 1215–1218. https://doi.org/10.1103/PhysRevLett.81.1215.
[13] Optics Communications 1987, 64 (4), 393–397. https://doi.org/10.1016/0030-4018(87)90258-6.
[14] Science Advances 2020, 6 (34), eabb5375. https://doi.org/10.1126/sciadv.abb5375.
[15] Long, The Raman Effect; John Wiley & Sons, Ltd, 2002;
[16] IEEE Journal of Quantum Electronics, vol. 24, no. 10, pp. 2076–2080, Oct. 1988, doi: 10.1109/3.8545.
[17] Journal of Raman Spectroscopy, vol. 22, no. 11, pp. 607–611, 1991, doi: 10.1002/jrs.1250221103.
[18] Hollenbeck and Cantrell, JOSA B 19, 2886-2892 (2002). https://doi.org/10.1364/JOSAB.19.002886
"""
function raman_parameters(material)
    if material == :N2
        rp = (kind = :molecular,
              rotation = :nonrigid,
              vibration = :sdo,
              B = 199.0, # [4]
              D = 5.74e-4, # [4]
              qJodd = 1, # [14] uses 3 as does [15]
              qJeven = 2, # [14] uses 6 as does [15]
              Δα = 6.7e-31, # [2] # note [14] use 1.86e-30
              # Bρr has a moderate dependence on J, which we ignore for now, taking J=8
              # [8] measured Bρr from 7 to 43 atm to be ~80e-3 cm^-1/atm,
              # which is translated to Hz/amg via
              # 80e-3*29979245800.0/(density(:N2, atm/bar)/amg)
              # giving ~2.6e9 Hz/amagat.
              # [7] gives ~3.3e9 Hz/amagat (measured at lower pressures), but they claim
              # their results are more accurate (of course!)
              Bρr = 3.3e9, # [7]
              Aρr = 0.0, # [7]
              dαdQ = 1.75e-20, # [6]
              Ωv = 2*π*2330.0*100.0*c, # [4]
              μ = (m_u*14.0067)^2/(2*m_u*14.0067),
              # For τ2v, [9] suggests pressure dependence is extremely weak up to 120 bar
              # [9] gives ~ 1.8 cm^-1, whereas Fig. 1 in [5] suggests something similar.
              # 1.8 cm^-1 = 0.054 THz
              # This gives a τ2v = 1/πΔν ~ 6 ps
              τ2v = 6e-12, # [5,9]
              )
    elseif material == :H2
        rp = (kind = :molecular,
              rotation = :nonrigid,
              vibration = :sdo,
              B = 5890.0, # [3]
              D = 5.0, # [3]
              qJodd = 3,
              qJeven = 1,
              Δα = 3e-31, # [3]
              Bρr = 114e6, # [7]
              Aρr = 6.15e6, # [7]
              dαdQ = 1.3e-20, # [3]
              Ωv = 2*π*124.5669e12,
              μ = (m_u*1.00784)^2/(2*m_u*1.00784),
              Bρv = 52.2e6, # [10]
              Aρv = 309e6, # [10]
              )
    elseif material == :D2
        rp = (kind = :molecular,
              rotation = :nonrigid,
              vibration = :sdo,
              B = 2930.0, # [3]
              D = 2.1, # [3]
              qJodd = 1,
              qJeven = 2,
              Δα = 3e-31, # [3]
              Bρr = 4e-3*100.0*c, # converted from [17], for J=0.
              Aρr = 0.0, # no data for this
              dαdQ = 1.4e-20, # [3]
              Ωv = 2*π*2987*100.0*c, # [11]
              μ = (m_u*2.014)^2/(2*m_u*2.014),
              Bρv = 120e6, # [16]
              Aρv = 101e6, # [16]
              )
    elseif material == :O2
        rp = (kind = :molecular,
              rotation = :nonrigid,
              vibration = :sdo,
              B = 144.0, # [2]
              D = 0.0, # TODO
              qJodd = 1,
              qJeven = 0,
              Δα = 10.2e-31, # [2]
              # TODO τ2r = 
              dαdQ = 1.46e-20, # [1]
              Ωv = 3e14, # [1]
              μ = 1.3e-26, # [1]
              # TODO τ2v = 
              )
    elseif material == :N2O
        rp = (kind = :molecular,
              rotation = :nonrigid,
              vibration = :none, # TODO work out correct parameters here
              B = 41.0, # [2]
              D = 0.0, # TODO
              qJodd = 1, # [14]
              qJeven = 1, # [14]
              Δα = 28.1e-31, # [2] note that [14] uses twice this
              τ2r = 23.8e-12, # [14]
              # TODO dαdQ =  
              Ωv = 2*π*1285*100.0*c,
              # TODO μ = 
              # TODO τ2v = 
             )
    elseif material == :SiO2 # [18]
        rp = (kind = :intermediate,
              K = 1.0,
              Ω = 1.0/12.2e-15,
              τ2 = 32e-15,
              ωi = 200 .*π.*c.*[56.25, 100.0, 231.25, 362.50, 463.00, 497.00, 611.50, 691.67, 793.67, 835.50, 930.0, 1080.00, 1215.00],
              Ai = [1.0, 11.40, 36.67, 67.67, 74.00, 4.50, 6.80, 4.60, 4.20, 4.50, 2.70, 3.10, 3.00],
              Γi = 100 .*π.*c.*[52.10, 110.42, 175.00, 162.50, 135.33, 24.50, 41.50, 155.0, 59.50, 64.30, 150.00, 91.00, 160.00],
              γi = 100 .*π.*c.*[17.37, 38.81, 58.33, 54.17, 45.11, 8.17, 13.83, 51.67, 19.83, 21.43, 50.00, 30.33, 53.33],
             )
    elseif material == :CH4
        rp = (kind = :molecular,
              rotation = :none,
              vibration = :sdo,
              dαdQ = 1.04e-20, # [6]
              Ωv = 2*π*2914*100.0*c, # [6]
              μ = (1.00784*m_u)/4, #(m_u*12.0107*m_u*1.00784)/(m_u*12.0107 + m_u*1.00784),
              Bρv = 384e6, # [16]
              Aρv = 0.0, # [16]
              Cv = 8220e6 # [16]
             )    
    elseif material == :SF6
        rp = (kind = :molecular,
                rotation = :none,
                vibration = :sdo,
                dαdQ = 1.23e-20, # [6]
                Ωv = 2*π*775*100.0*c, # [6]
                μ = (18.998403*m_u)/6,
                τ2v = 6.6e-12, # [13]
                )      
    else
        throw(DomainError(material, "Unknown material $material"))
    end
    rp
end

"""
    lookup_mirror(type)

Create a `CSpline` interpolant for the complex-valued reflectivity of a mirror of `type`.
"""
function lookup_mirror(type)
    if type == :PC70
        # λ (nm), R(5deg) (%), R(19deg) (%)
        Rdat = CSV.File(joinpath(Utils.datadir(), "PC70_R.csv"))
        # λ (nm), GDD(5deg) (fs^2), GDD(19deg) (fs^2)
        GDDdat = CSV.File(joinpath(Utils.datadir(), "PC70_GDD.csv"))
        # Double sqrt creates average reflectivity per _reflection_ rather than per pair
        rspl = Maths.BSpline(Rdat.Wlgth*1e-9, sqrt.(sqrt.(Rdat.Rp5deg/100 .* Rdat.Rp19deg/100)))
        λGDD = GDDdat.Wlgth
        ω = wlfreq.(λGDD*1e-9)
        # average phase per _reflection_ rather than per pair
        ϕ = 1e-30/2 * Maths.cumtrapz(Maths.cumtrapz(GDDdat.GDDrp5deg.+GDDdat.GDDrp19deg, ω), ω)
        # ϕ has a large linear component - remove that
        ωfs = ω*1e-15
        ωfs0 = wlfreq(800e-9)*1e-15
        idcs =  2 .< ωfs .< 4 # large kinks at edge of frequency window confuse the fit
        p = Polynomials.fit(ωfs[idcs] .- ωfs0, ϕ[idcs], 5)
        p[2:end] = 0 # polynomials use 0-based indexing - only use constant and linear term
        ϕ .-= p.(ωfs .- ωfs0) # subtract linear part
        ϕspl = Maths.BSpline(λGDD*1e-9, ϕ)
        return λ -> rspl(λ) * exp(-1im*ϕspl(λ)) * Maths.planck_taper(
            λ, 400e-9, 450e-9, 1200e-9, 1300e-9)
    elseif type == :HD59
        dat = readdlm(joinpath(Utils.datadir(), "HD59_GDD.dat"); skipstart=2)
        λ = dat[:, 1] * 1e-9
        ω = wlfreq.(λ)
        GDD = dat[:, 2] .* 1e-30
        ϕ = Maths.cumtrapz(Maths.cumtrapz(GDD, ω), ω)
        ωfs = ω*1e-15
        ωfs0 = wlfreq(1030e-9)*1e-15
        p = Polynomials.fit(ωfs .- ωfs0, ϕ, 5)
        p[2:end] = 0 # polynomials use 0-based indexing - only use constant and linear term
        ϕ .-= p.(ωfs .- ωfs0) # subtract linear part
        ϕspl = Maths.BSpline(λ, ϕ)
        return λ -> exp(-1im*ϕspl(λ)) * Maths.planck_taper(
            λ, 993e-9, 1000e-9, 1060e-9, 1075e-9)
    elseif type == :PC147
        dat = readdlm(joinpath(Utils.datadir(), "PC147.txt"); skipstart=1)
        λR = dat[:, 1] * 1e-9
        R = dat[:, 2] # average reflectivity per mirror (complementary pair)
        rspl = Maths.BSpline(λR, sqrt.(R/100))
        λGDD = dat[:, 3] * 1e-9
        ω = wlfreq.(λGDD)
        GDD = dat[:, 4] .* 1e-30 # average GDD per mirror (complementary pair)
        ϕ = Maths.cumtrapz(Maths.cumtrapz(GDD, ω), ω)
        ωfs = ω*1e-15
        ωfs0 = wlfreq(1030e-9)*1e-15
        p = Polynomials.fit(ωfs .- ωfs0, ϕ, 5)
        p[2:end] = 0 # polynomials use 0-based indexing - only use constant and linear term
        ϕ .-= p.(ωfs .- ωfs0) # subtract linear part
        ϕspl = Maths.BSpline(λGDD, ϕ)
        return λ -> rspl(λ) * exp(-1im*ϕspl(λ)) * Maths.planck_taper(
            λ, 640e-9, 650e-9, 1350e-9, 1360e-9)
    elseif type == :PC1611
        dat = readdlm(joinpath(Utils.datadir(), "PC1611.txt"); skipstart=1)
        λR = dat[:, 1] * 1e-9
        R = dat[:, 2] 
        rspl = Maths.BSpline(λR, sqrt.(R/100))
        λGDD = dat[:, 3] * 1e-9
        ω = wlfreq.(λGDD)
        GDD = dat[:, 4] .* 1e-30
        ϕ = Maths.cumtrapz(Maths.cumtrapz(GDD, ω), ω)
        ωfs = ω*1e-15
        ωfs0 = wlfreq(1030e-9)*1e-15
        p = Polynomials.fit(ωfs .- ωfs0, ϕ, 5)
        p[2:end] = 0 # polynomials use 0-based indexing - only use constant and linear term
        ϕ .-= p.(ωfs .- ωfs0) # subtract linear part
        ϕspl = Maths.BSpline(λGDD, ϕ)
        return λ -> rspl(λ) * exp(-1im*ϕspl(λ)) * Maths.planck_taper(
            λ, 850e-9, 855e-9, 1195e-9, 1200e-9)
    elseif type == :PC1821
        dat = readdlm(joinpath(Utils.datadir(), "PC1821.txt"); skipstart=1)
        λR = dat[:, 1] * 1e-9
        R = dat[:, 2] 
        rspl = Maths.BSpline(λR, sqrt.(R/100))
        λGDD = dat[:, 3] * 1e-9
        ω = wlfreq.(λGDD)
        GDD = dat[:, 4] .* 1e-30
        ϕ = Maths.cumtrapz(Maths.cumtrapz(GDD, ω), ω)
        ωfs = ω*1e-15
        ωfs0 = wlfreq(1030e-9)*1e-15
        p = Polynomials.fit(ωfs .- ωfs0, ϕ, 5)
        p[2:end] = 0 # polynomials use 0-based indexing - only use constant and linear term
        ϕ .-= p.(ωfs .- ωfs0) # subtract linear part
        ϕspl = Maths.BSpline(λGDD, ϕ)
        return λ -> rspl(λ) * exp(-1im*ϕspl(λ)) * Maths.planck_taper(
            λ, 800e-9, 805e-9, 1345e-9, 1350e-9)
    elseif type == :HD120
        dat = readdlm(joinpath(Utils.datadir(), "HD120.csv"), ','; skipstart=1)
        λR = dat[:, 1] * 1e-9
        R = dat[:, 2] # reflectivity per mirror 
        rspl = Maths.BSpline(λR, sqrt.(R/100))
        λGDD = dat[:, 3] * 1e-9
        ω = wlfreq.(λGDD)
        GDD = dat[:, 4] .* 1e-30 # GDD per mirror
        ϕ = Maths.cumtrapz(Maths.cumtrapz(GDD, ω), ω)
        ωfs = ω*1e-15
        ωfs0 = wlfreq(1030e-9)*1e-15
        p = Polynomials.fit(ωfs .- ωfs0, ϕ, 5)
        p[2:end] = 0 # polynomials use 0-based indexing - only use constant and linear term
        ϕ .-= p.(ωfs .- ωfs0) # subtract linear part
        ϕspl = Maths.BSpline(λGDD, ϕ)
        return λ -> rspl(λ) * exp(-1im*ϕspl(λ)) * Maths.planck_taper(
            λ, 880e-9, 900e-9, 1200e-9, 1220e-9)
    elseif type == :ThorlabsUMC
        # λ (nm), R(p) (%), R(s) (%)
        Rdat = CSV.File(joinpath(Utils.datadir(), "UCxx-15FS_R.csv"))
        # λ (nm), GD(p) (fs^2), GD(s) (fs^2)
        GDdat = CSV.File(joinpath(Utils.datadir(), "UCxx-15FS_GD.csv"))
        # Default to s-pol
        rspl = Maths.BSpline(Rdat.wl*1e-9, sqrt.(Rdat.Rs/100))
        λGD = GDdat.wl
        ω = wlfreq.(λGD*1e-9)
        # average phase per reflection, default to s-pol
        ϕ = Maths.cumtrapz(1e-15*GDdat.GDs, ω)
        # ϕ has a large linear component - remove that
        ωfs = ω*1e-15
        ωfs0 = wlfreq(800e-9)*1e-15
        idcs =  2 .< ωfs .< 4 # large kinks at edge of frequency window confuse the fit
        p = Polynomials.fit(ωfs[idcs] .- ωfs0, ϕ[idcs], 3)
        p[2:end] = 0 # polynomials use 0-based indexing - only use constant and linear term
        ϕ .-= p.(ωfs .- ωfs0) # subtract linear part
        ϕspl = Maths.BSpline(λGD*1e-9, ϕ)
        return λ -> rspl(λ) * exp(-1im*ϕspl(λ)) * Maths.planck_taper(
            λ, 640e-9, 650e-9, 1050e-9, 1100e-9)
    else
        throw(DomainError("Unknown mirror type $type"))
    end
end

"""
    process_mirror_data(λR, R, λGDD, GDD, λ0, λmin, λmax; fitorder=5, windowwidth=20e-9)

Process reflectivity and group-delay dispersion data for a mirror and create a transfer function for
a frequency-domain electric field representing the mirror.

# Arguments:
- `λR`: wavelength samples for reflectivity in SI units (m)
- `R`: mirror reflectivity (between 0 and 1)
- `λGDD`: wavelength samples for GDD in SI units (m)
- `GDD`: GDD in SI units (s²)
- `λ0`: central wavelength (used to remove any overall group delay)
- `λmin`, `λmax`: bounds of the wavelength region to apply the transfer function over

# Keyword arguments
- `fitorder`: order of polynomial fit to use in removing overall group delay (default: 5)
- `windowwidth`: wavelength width of the smoothing region outside `(λmin, λmax)`
                for the window in SI units (default: 20e-9, i.e. 20 nm)
"""
function process_mirror_data(λR, R, λGDD, GDD, λ0, λmin, λmax; fitorder=5, windowwidth=20e-9)
    r = sqrt.(R)
    rspl = Maths.BSpline(λR, r)
    ω = wlfreq.(λGDD)
    ϕ = Maths.cumtrapz(Maths.cumtrapz(GDD, ω), ω)
    ωfs = ω*1e-15
    ωfs0 = wlfreq(λ0)*1e-15
    p = Polynomials.fit(ωfs .- ωfs0, ϕ, fitorder)
    p[2:end] = 0 # polynomials use 0-based indexing - only use constant and linear term
    ϕ .-= p.(ωfs .- ωfs0) # subtract linear part
    ϕspl = Maths.BSpline(λGDD, ϕ)
    return λ -> rspl(λ) * exp(-1im*ϕspl(λ)) * Maths.planck_taper(
        λ, λmin-windowwidth, λmin, λmax, λmax+windowwidth)
end

end