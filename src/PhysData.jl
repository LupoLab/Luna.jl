module PhysData

import CoolProp
import PhysicalConstants: CODATA2014
import Unitful: ustrip
import CSV
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

const gas = (:Air, :He, :HeJ, :Ne, :Ar, :Kr, :Xe, :N2, :H2)
const gas_str = Dict(
    :He => "He",
    :HeJ => "He",
    :Ar => "Ar",
    :Ne => "Neon",
    :Kr => "Krypton",
    :Xe => "Xenon",
    :Air => "Air",
    :N2 => "Nitrogen",
    :H2 => "Hydrogen"
)
const glass = (:SiO2, :BK7, :KBr, :CaF2, :BaF2, :Si, :MgF2, :ADPo, :ADPe, :KDPo, :KDPe)
const metal = (:Ag,:Al)

"Change from ω to λ and vice versa"
wlfreq(ωλ) = 2π*c/ωλ

"convert Δλ at λ to Δω"
ΔλΔω(Δλ, λ) = (2π*c)*Δλ/λ^2

eV_to_m(eV) = wlfreq(electron*eV/ħ)

"Linear coefficients"

"Sellmeier expansion for linear susceptibility from Applied Optics 47, 27, 4856 (2008) at
room temperature and atmospheric pressure"
function γ_Börzsönyi(B1, C1, B2, C2)
    return μm -> (B1 * μm^2 / (μm^2 - C1) + B2 * μm^2 / (μm^2 - C2))
end

"Adapted Sellmeier expansion for helium made to fit high frequency data
Phys. Rev. A 92, 033821 (2015)"
function γ_JCT(B1, C1, B2, C2, B3, C3)
    return μm -> (B1 * μm^2 / (μm^2 - C1)
                  + B2 * μm^2 / (μm^2 - C2)
                  + B3 * μm^2 / (μm^2 - C3))
end

"
Sellmeier expansion for linear susceptibility from
J. Opt. Soc. Am. 67, 1550 (1977)
"
function γ_Peck(B1, C1, B2, C2, dens)
    return μm -> @. (((B1 / (C1 - 1/μm^2) + B2 / (C2 - 1/μm^2)) + 1)^2 - 1)/dens
end

"Sellemier expansion for gases. Return function for linear polarisability γ, i.e.
susceptibility of a single particle."
function sellmeier_gas(material::Symbol)
    dens = density(material, 1.0, 273.15)
    if material == :He
        B1 = 4977.77e-8
        C1 = 28.54e-6
        B2 = 1856.94e-8
        C2 = 7.76e-3
        return γ_Börzsönyi(B1/dens, C1, B2/dens, C2)
    elseif material == :HeJ
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
    elseif material == :H2
        B1 = 14895.6e-6
        C1 = 180.7
        B2 = 4903.7e-6
        C2 = 92.0
        return γ_Peck(B1, C1, B2, C2, density(material, atm/bar, 273.15))
    else
        throw(DomainError(material, "Unknown gas $material"))
    end
end

"Sellmeier for glasses. Returns function of wavelength in μm which in turn
returns the refractive index directly"
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

"Get function to return χ1 as a function of:
    wavelength in SI units
    pressure in bar
    temperature in Kelvin
Gases only."
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

"Get χ1 at wavelength λ in SI units, pressure P in bar and temperature T in Kelvin.
Gases only."
function χ1(gas::Symbol, λ, P=1.0, T=roomtemp)
    return χ1_fun(gas)(λ, P, T)
end


"Get refractive index for any material at wavelength given in SI units"
function ref_index(material, λ, P=1.0, T=roomtemp; lookup=nothing)
    return ref_index_fun(material, P, T; lookup=lookup)(λ)
end

"Get function which returns refractive index."
function ref_index_fun(material::Symbol, P=1.0, T=roomtemp; lookup=nothing)
    if material in gas
        χ1 = χ1_fun(material, P, T)
        return λ -> sqrt(1 + χ1(λ))
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

"Get function which returns refractive index for gas mixture."
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

"Get function which returns ref index for mixture as function of wavelength and densities."
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

"Get reflection coefficients"
function fresnel(n2, θi; n1=1.0)
    θt = asin(n1*sin(θi)/n2)
    rs = (n1*cos(θi) - n2*cos(θt))/(n1*cos(θi) + n2*cos(θt))
    rp = (n2*cos(θi) - n1*cos(θt))/(n2*cos(θi) + n1*cos(θt))
    abs2(rs), angle(rs), abs2(rp), angle(rp)
end


"Nonlinear coefficients"

"Calculate single-molecule third-order hyperpolarisability of a gas
at given wavelength(s) and at room temperature.
If source == :Bishop:
Uses reference values to calculate γ
If source == :Lehmeier (default):
Uses scaling factors to calculate χ3 at 1 bar and scales by density
to get to a single molecule i.e. the hyperpolarisability

References:
[1] Journal of Chemical Physics, AIP, 91, 3549-3551 (1989)
[2] Chemical Reviews, 94, 3-29 (1994)
[3] Optics Communications, 56(1), 67–72 (1985)
[4] Phys. Rev. A, vol. 42, 2578 (1990)

TODO: More Bishop/Shelton; Wahlstrand updated values.

"
function γ3_gas(material::Symbol; source=nothing)
    if source == nothing
        if material in (:He, :HeJ, :Ne, :Ar, :Kr, :Xe, :N2)
            source = :Lehmeier
        elseif material in (:H2,)
            source = :Shelton
        else
            error("no default γ3 source for material: $material")
        end
    end
    dens = density(material, atm/bar, 273.15)
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
        elseif material == :N2
            fac = 21.1
        else
            throw(DomainError(material, "Lehmeier model does not include $material"))
        end
        return 4*fac*3.43e-28 / dens
    elseif source == :Shelton
        # ref [4]
        if material == :H2
            return 2.2060999099841444e-26 / dens # TODO: check this carefully
        else
            throw(DomainError(material, "Shelton model does not include $material"))
        end
    else
        throw(DomainError(source, "Unkown γ3 model $source"))
    end
end

function χ3_gas(material::Symbol, P, T=roomtemp; source=nothing)
    return γ3_gas(material, source=source) .* density.(material, P, T)
end

function n2_gas(material::Symbol, P, T=roomtemp, λ=800e-9; source=nothing)
    n0 = ref_index(material, λ, P, T)
    return @. 3/4 * χ3_gas(material, P, T, source=source) / (ε_0*c*n0^2)
end

"""
    density(gas::Symbol, P, T=roomtemp)

Number density of `gas` [m^-3] at pressure `P` [bar] and temperature `T` [K].
"""
function density(gas::Symbol, P, T=roomtemp)
    P == 0 ? zero(P) : CoolProp.PropsSI("DMOLAR", "T", T, "P", bar*P, gas_str[gas])*N_A
end

function pressure(gas, density, T=roomtemp)
    density == 0 ? zero(density) :
                   CoolProp.PropsSI("P", "T", T, "DMOLAR", density/N_A, gas_str[gas])/bar
end

function densityspline(gas::Symbol; Pmax, Pmin=0, N=2^10, T=roomtemp)
    P = collect(range(Pmin, Pmax, length=N))
    ρ = density.(gas, P, T)
    Maths.CSpline(P, ρ)
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
    elseif material == :N2
        Ip = 0.5726
    elseif material == :H2
        Ip = 0.5669
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
    elseif material == :Ne
        return 2, 1, 1;
    elseif material == :Kr
        return 4, 1, 1
    elseif material == :Xe
        return 5, 1, 1
    elseif material in (:He, :HeJ)
        return 1, 0, 1
    else
        throw(DomainError(material, "Unknown material $material"))
    end
end

function lookup_glass(material::Symbol)
    if material == :SiO2
        data = data_glass(:SiO2)
        spl = Maths.BSpline(1e6*eV_to_m.(data[:, 1]), data[:, 2] + 1im * data[:, 3])
    else
        throw(DomainError(material, "Unknown metal $material"))
    end
    return spl
end

"Returns function of wavelength in μm which in turn
 returns the refractive index directly"
function lookup_metal(material::Symbol)
    data = data_metal(material)::Array{Float64,2}
    nspl = Maths.BSpline(data[:,1], data[:,2] .+ im.*data[:,3])
    return nspl
end

"""
Get the Raman parameters for `material`.

# Fields
Fields in the returned named tuple must include:
- `kind::Symbol`: one of `:molecular` or ...

If `kind == :molecular` then the following must also be specified:
- `rotation::Symbol`: only `:nonrigid` or `:none` supported at present.
- `vibration::Symbol`: only `:sdo` or `:none` supported at present.

If `rotation == :nonrigid` then the following must also be specified:
- `B::Real`: the rotational constant [1/m]
- `Δα::Real`: molecular polarizability anisotropy [m^3]
- `τ2r::Real`: coherence time [s]
- `qJodd::Integer`: nuclear spin parameter for odd `J`
- `qJeven::Integer`: nuclear spin parameter for even `J`
- `D::Real=0.0`: centrifugal constant [1/m]

If `vibration == :sdo` then the following must also be specified:
- `Ωv::Real`: vibrational frequency [rad/s]
- `dαdQ::Real`: isotropic averaged polarizability derivative [m^2]
- `μ::Real`: reduced molecular mass [kg]
- `τ2v::Real`: coherence time [s]

# References
[1] Phys. Rev. A, 94, 023816 (2016)
[2] Phys. Rev. A, 85, 043820 (2012)
[3] Phys. Rev. A, 92, 063828 (2015)
[4] Journal of Raman Spectroscopy 2, 133 (1974)
[5] J. Phys. Chem., 91, 41 (1987)
[6] Applied Spectroscopy 23, 211 (1969)
[7] Phys. Rev. A, 34, 3, 1944 (1986)
"""
function raman_parameters(material)
    if material == :N2
        rp = (kind = :molecular,
              rotation = :nonrigid,
              vibration = :sdo,
              B = 199.0, # [4]
              D = 5.74e-4, # [4]
              qJodd = 1,
              qJeven = 2,
              Δα = 6.7e-31, # [2]
              τ2r = 2e-12, # [7] TODO pressure dependence
              dαdQ = 1.75e-20, # [6]
              Ωv = 2*π*2330.0*100.0*c, # [4]
              μ = 1.16e-26,
              τ2v = 8.8e-12, # [5] TODO pressure dependence
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
              τ2r = 280e-12, # at 10 bar, TODO pressure dependence
              dαdQ = 1.3e-20, # [3]
              Ωv = 2*π*124.5669e12,
              μ = 8.369e-28,
              τ2v = 578e-12, # at 10 bar, TODO pressure dependence
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
              # TODO τ2r = 
              dαdQ = 1.4e-20, # [3]
              # TODO Ωv = 
              # TODO μ = 
              # TODO τ2v = 
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
              vibration = :sdo,
              B = 41.0, # [2]
              D = 0.0, # TODO
              # TODO qJodd = 
              # TODO qJeven = 
              Δα = 28.1e-31, # [2]
              # TODO τ2r = 
              # TODO dαdQ =  
              # TODO Ωv =  
              # TODO μ = 
              # TODO τ2v = 
             )           
    else
        throw(DomainError(material, "Unknown material $material"))
    end
    rp
end

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

end