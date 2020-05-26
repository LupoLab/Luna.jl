module PhysData

import CoolProp
import PhysicalConstants: CODATA2014
import Unitful: ustrip
import CSV
import Luna: Maths, Utils

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
const glass = (:SiO2, :BK7, :KBr, :CaF2, :BaF2, :Si, :MgF2)
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
    else
        throw(DomainError(material, "Unknown glass $material"))
    end
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
function ref_index(material::Symbol, λ, P=1.0, T=roomtemp; lookup=nothing)
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
    elseif material in (:He, :HeJ)
        return 1, 1, 1
    else
        throw(DomainError(material, "Unknown material $material"))
    end
end

function lookup_glass(material::Symbol)
    if material == :SiO2
        ndat = CSV.read(joinpath(Utils.datadir(), "silica_n.csv"))
        kdat = CSV.read(joinpath(Utils.datadir(), "silica_k.csv"))
        spl = Maths.BSpline(1e6*eV_to_m.(ndat[:, 1]), ndat[:, 2] + 1im * kdat[:, 2])
    else
        throw(DomainError(material, "Unknown metal $material"))
    end
    return spl
end

"Lookup tables for complex refractive indices of metals."
function data_metal(material::Symbol)
            # Below: 0.127: W. S. M. Werner, K. Glantschnig, C. Ambrosch-Draxl.
            # Optical constants and inelastic electron-scattering data for 17 elemental metals,
            # J. Phys Chem Ref. Data 38, 1013-1092 (2009)
            # Above 0.206: S. Babar and J. H. Weaver.
            # Optical constants of Cu, Ag, and Au revisited, Appl. Opt. 54, 477-481 (2015)
            # Below 0.206: K. Stahrenberg, Th. Herrmann, K. Wilmers, N. Esser, W. Richter, and M. J. G. Lee.
            # Optical properties of copper and silver in the energy range 2.5-9.0 eV,
            # Phys Rev. B 64, 115111 (2001) (Numerical data kindly provided by Prof. Dr. Norbert Esser)
            # above 12.4: H.-J. Hagemann, W. Gudat, and C. Kunz.
            # Optical constants from the far infrared to the x-ray region: Mg, Al, Cu, Ag, Au, Bi, C, and Al2O3,
            # J. Opt. Soc. Am. 65, 742-744 (1975)
            dat = ( Ag = Float64[
                0.00236 0.99667 0.00774;
                0.00316 0.99770 0.00530;
                0.00447 0.99198 0.00237;
                0.00676 0.9813 0.00291;
                0.01140 0.9172 0.00818;
                0.01355 0.860 0.0423;
                0.01714 0.863 0.293;
                0.02430 0.928 0.310;
                0.02563 0.927 0.346;
                0.03038 0.895 0.302;
                0.03270 0.894 0.304;
                0.03568 0.872 0.351;
                0.03793 0.896 0.427;
                0.04059 0.953 0.451;
                0.04478 0.889 0.425;
                0.04607 0.848 0.446;
                0.04895 0.833 0.507;
                0.05391 1.04 0.746;
                0.05843 1.09 0.658;
                0.06163 1.06 0.564;
                0.06714 1.03 0.586;
                0.07185 0.982 0.596;
                0.07359 1.13 0.751;
                0.07437 1.17 0.721;
                0.08345 1.28 0.750;
                0.08780 1.37 0.688;
                0.09198 1.32 0.657;
                0.09321 1.27 0.625;
                0.09888 1.32 0.586;
                0.10258 1.23 0.571;
                0.10482 1.27 0.577;
                0.10667 1.30 0.565;
                0.10857 1.30 0.550;
                0.11350 1.33 0.518;
                0.12007 1.27 0.490;
                0.12157 1.21 0.515;
            0.12782 1.10047 0.72025;
            0.12808 1.10131 0.72092;
            0.12835 1.10177 0.72135;
            0.12861 1.10169 0.72134;
            0.12888 1.10100 0.72080;
            0.12915 1.09967 0.71975;
            0.12942 1.09762 0.71822;
            0.12969 1.09481 0.71619;
            0.12996 1.09145 0.71374;
            0.13024 1.08796 0.71109;
            0.13051 1.08474 0.70849;
            0.13078 1.08194 0.70609;
            0.13106 1.07956 0.70392;
            0.13134 1.07761 0.70203;
            0.13162 1.07600 0.70043;
            0.13190 1.07450 0.69911;
            0.13218 1.07305 0.69803;
            0.13246 1.07165 0.69717;
            0.13275 1.07019 0.69646;
            0.13303 1.06863 0.69586;
            0.13332 1.06700 0.69539;
            0.13360 1.06519 0.69501;
            0.13389 1.06311 0.69468;
            0.13418 1.06069 0.69431;
            0.13447 1.05788 0.69384;
            0.13477 1.05470 0.69326;
            0.13506 1.05129 0.69264;
            0.13535 1.04786 0.69208;
            0.13565 1.04452 0.69169;
            0.13595 1.04136 0.69159;
            0.13625 1.03835 0.69186;
            0.13655 1.03546 0.69254;
            0.13685 1.03258 0.69357;
            0.13715 1.02959 0.69486;
            0.13745 1.02649 0.69631;
            0.13776 1.02333 0.69786;
            0.13807 1.02008 0.69935;
            0.13838 1.01667 0.70070;
            0.13868 1.01312 0.70195;
            0.13900 1.00959 0.70326;
            0.13931 1.00630 0.70479;
            0.13962 1.00327 0.70659;
            0.13994 1.00043 0.70860;
            0.14025 0.99765 0.71068;
            0.14057 0.99478 0.71266;
            0.14089 0.99170 0.71446;
            0.14121 0.98833 0.71602;
            0.14153 0.98477 0.71742;
            0.14186 0.98106 0.71871;
            0.14218 0.97724 0.71992;
            0.14251 0.97339 0.72109;
            0.14284 0.96943 0.72222;
            0.14317 0.96532 0.72333;
            0.14350 0.96113 0.72452;
            0.14383 0.95706 0.72591;
            0.14417 0.95333 0.72764;
            0.14450 0.95000 0.72972;
            0.14484 0.94698 0.73206;
            0.14518 0.94414 0.73458;
            0.14552 0.94137 0.73720;
            0.14586 0.93855 0.73982;
            0.14621 0.93563 0.74239;
            0.14655 0.93257 0.74486;
            0.14690 0.92931 0.74718;
            0.14725 0.92578 0.74931;
            0.14760 0.92200 0.75125;
            0.14795 0.91810 0.75314;
            0.14831 0.91416 0.75509;
            0.14866 0.91017 0.75713;
            0.14902 0.90608 0.75929;
            0.14938 0.90197 0.76166;
            0.14974 0.89796 0.76437;
            0.15010 0.89411 0.76744;
            0.15047 0.89043 0.77090;
            0.15083 0.88696 0.77474;
            0.15120 0.88370 0.77897;
            0.15157 0.88061 0.78354;
            0.15194 0.87762 0.78837;
            0.15231 0.87469 0.79342;
            0.15269 0.87180 0.79864;
            0.15307 0.86889 0.80393;
            0.15345 0.86587 0.80921;
            0.15383 0.86267 0.81443;
            0.15421 0.85933 0.81961;
            0.15459 0.85590 0.82477;
            0.15498 0.85246 0.82997;
            0.15537 0.84911 0.83528;
            0.15576 0.84597 0.84077;
            0.15615 0.84313 0.84652;
            0.15655 0.84073 0.85258;
            0.15694 0.83879 0.85897;
            0.15734 0.83725 0.86563;
            0.15774 0.83597 0.87247;
            0.15814 0.83480 0.87934;
            0.15855 0.83363 0.88619;
            0.15895 0.83246 0.89296;
            0.15936 0.83130 0.89969;
            0.15977 0.83023 0.90640;
            0.16019 0.82930 0.91312;
            0.16060 0.82853 0.91987;
            0.16102 0.82797 0.92667;
            0.16144 0.82762 0.93353;
            0.16186 0.82750 0.94043;
            0.16228 0.82762 0.94737;
            0.16271 0.82798 0.95434;
            0.16314 0.82857 0.96136;
            0.16357 0.82933 0.96839;
            0.16400 0.83021 0.97545;
            0.16444 0.83118 0.98251;
            0.16487 0.83223 0.98958;
            0.16531 0.83088 0.99959;
            0.16575 0.83127 1.01158;
            0.16620 0.82727 1.02560;
            0.16665 0.82808 1.03261;
            0.16709 0.82886 1.03820;
            0.16755 0.82962 1.04289;
            0.16800 0.83035 1.04727;
            0.16846 0.83108 1.05195;
            0.16892 0.83186 1.05693;
            0.16938 0.83273 1.06215;
            0.16984 0.83368 1.06808;
            0.17031 0.83476 1.07462;
            0.17078 0.83595 1.08125;
            0.17125 0.83723 1.08787;
            0.17172 0.83860 1.09421;
            0.17220 0.84003 1.10000;
            0.17268 0.84149 1.10533;
            0.17316 0.84298 1.11071;
            0.17365 0.84453 1.11637;
            0.17413 0.84615 1.12232;
            0.17463 0.84786 1.12868;
            0.17512 0.84966 1.13554;
            0.17561 0.85155 1.14216;
            0.17611 0.85353 1.14846;
            0.17662 0.85559 1.15476;
            0.17712 0.85774 1.16112;
            0.17763 0.85997 1.16755;
            0.17814 0.86228 1.17387;
            0.17865 0.86465 1.18014;
            0.17917 0.86708 1.18656;
            0.17969 0.86952 1.19291;
            0.18021 0.87199 1.19907;
            0.18073 0.87449 1.20500;
            0.18126 0.87706 1.21088;
            0.18179 0.87969 1.21643;
            0.18233 0.88237 1.22161;
            0.18287 0.88512 1.22669;
            0.18341 0.88797 1.23201;
            0.18395 0.89094 1.23792;
            0.18450 0.89402 1.24420;
            0.18505 0.89720 1.25057;
            0.18560 0.90045 1.25677;
            0.18616 0.90374 1.26250;
            0.18672 0.90705 1.26788;
            0.18729 0.91045 1.27341;
            0.18785 0.91394 1.27929;
            0.18843 0.91750 1.28535;
            0.18900 0.92111 1.29122;
            0.18958 0.92475 1.29654;
            0.19016 0.92845 1.30169;
            0.19074 0.93225 1.30693;
            0.19133 0.93614 1.31227;
            0.19193 0.94009 1.31770;
            0.19252 0.94411 1.32307;
            0.19312 0.94815 1.32826;
            0.19373 0.95224 1.33342;
            0.19433 0.95637 1.33861;
            0.19494 0.96053 1.34372;
            0.19556 0.96469 1.34858;
            0.19618 0.96886 1.35307;
            0.19680 0.97303 1.35722;
            0.19743 0.97727 1.36115;
            0.19806 0.98163 1.36507;
            0.19869 0.98618 1.36935;
            0.19933 0.99098 1.37427;
            0.19997 0.99599 1.37936;
            0.20062 1.00116 1.38449;
            0.20127 1.00650 1.38965;
            0.20193 1.01200 1.39469;
            0.20259 1.01760 1.39932;
            0.20325 1.02330 1.40363;
            0.20392 1.02910 1.40767;
        0.2066 1.079 1.247;
        0.2101 1.101 1.258;
        0.2138 1.121 1.267;
        0.2175 1.140 1.272;
        0.2214 1.157 1.280;
        0.2254 1.169 1.283;
        0.2296 1.181 1.287;
        0.2339 1.190 1.294;
        0.2384 1.198 1.306;
        0.2431 1.215 1.317;
        0.2480 1.233 1.326;
        0.2530 1.254 1.328;
        0.2583 1.270 1.327;
        0.2638 1.294 1.329;
        0.2695 1.321 1.310;
        0.2755 1.348 1.291;
        0.2818 1.380 1.247;
        0.2883 1.402 1.181;
        0.2952 1.427 1.079;
        0.3024 1.410 0.894;
        0.3100 1.265 0.656;
        0.3179 0.859 0.466;
        0.3263 0.307 0.651;
        0.3351 0.187 1.042;
        0.3444 0.142 1.269;
        0.3542 0.106 1.467;
        0.3647 0.076 1.645;
        0.3757 0.061 1.815;
        0.3875 0.050 1.981;
        0.3999 0.054 2.138;
        0.4133 0.050 2.292;
        0.4275 0.051 2.448;
        0.4428 0.052 2.604;
        0.4592 0.052 2.765;
        0.4769 0.053 2.930;
        0.4959 0.052 3.105;
        0.5166 0.052 3.288;
        0.5391 0.05168 3.483;
        0.5636 0.05009 3.694;
        0.5904 0.04977 3.918;
        0.6199 0.04803 4.164;
        0.6525 0.04964 4.432;
        0.6888 0.05080 4.725;
        0.7293 0.05151 5.048;
        0.7749 0.05269 5.409;
        0.8266 0.05504 5.814;
        0.8856 0.05896 6.276;
        0.9537 0.06542 6.802;
        1.033 0.07286 7.412;
        1.127 0.08427 8.128;
        1.240 0.09799 8.981;
        1.305 0.1056 9.472;
        1.378 0.1143 10.018;
        1.459 0.1256 10.63;
        1.550 0.1388 11.31;
        1.653 0.1543 12.08;
        1.771 0.1735 12.97;
        1.907 0.1988 13.98;
        2.066 0.2315 15.16;
        2.254 0.2724 16.56;
        2.480 0.3256 18.23;
        2.755 0.3970 20.26;
        3.100 0.4960 22.80;
        3.542 0.6387 26.05;
        4.133 0.8566 30.37;
        4.959 1.228 36.37;
        6.199 1.851 45.26;
        8.266 3.227 59.73;
        12.4000 5.079 86.53;
        2.480E+01 3.670E+01 1.73E+02;
        4.959E+01 1.180E+02 3.06E+02;
        1.240E+02 3.090E+02 5.06E+02;
        2.480E+02 5.310E+02 6.89E+02
        ],
        # Below 0.14: H.-J. Hagemann, W. Gudat, and C. Kunz.
        # Optical constants from the far infrared to the x-ray region: Mg, Al, Cu, Ag, Au, Bi, C, and Al2O3,
        # J. Opt. Soc. Am. 65, 742-744 (1975)
        # K. M. McPeak, S. V. Jayanti, S. J. P. Kress, S. Meyer, S. Iotti, A. Rossinelli, and D. J. Norris.
        # Plasmonic films can easily be better: Rules and recipes, ACS Photonics 2, 326-333 (2015)
        # Above 2.0: M. A. Ordal, R. J. Bell, R. W. Alexander, L. A. Newquist, M. R. Querry.
        # Optical properties of Al, Fe, Ti, Ta, W, and Mo at submillimeter wavelengths,
        # Appl. Opt. 27, 1203-1209 (1988)
        Al = Float64[
            1.033E-05 9.990E-01 8.24E-12;
            1.240E-05 1.001E+00 9.88E-12;
            1.550E-05 1.000E+00 2.47E-11;
            2.066E-05 1.001E+00 8.09E-11;
            2.480E-05 1.001E+00 1.50E-10;
            3.100E-05 1.002E+00 3.77E-10;
            4.133E-05 1.002E+00 1.06E-09;
            6.199E-05 1.001E+00 4.68E-09;
            8.266E-05 1.000E+00 1.46E-08;
            1.127E-04 1.002E+00 5.14E-08;
            1.378E-04 1.000E+00 1.12E-07;
            1.771E-04 1.003E+00 2.97E-07;
            2.480E-04 1.002E+00 1.05E-06;
            4.133E-04 1.001E+00 7.51E-06;
            6.199E-04 1.000E+00 3.24E-05;
            7.293E-04 1.000E+00 6.13E-05;
            7.514E-04 1.000E+00 4.62E-05;
            7.749E-04 1.000E+00 2.32E-05;
            7.999E-04 1.000E+00 6.43E-06;
            8.266E-04 1.000E+00 7.14E-06;
            1.240E-03 1.000E+00 3.10E-05;
            1.550E-03 1.000E+00 7.52E-05;
            2.066E-03 9.990E-01 2.27E-04;
            2.480E-03 9.990E-01 4.66E-04;
            3.100E-03 9.980E-01 1.01E-03;
            3.542E-03 9.960E-01 1.54E-03;
            4.133E-03 9.960E-01 2.57E-03;
            4.428E-03 9.960E-01 3.31E-03;
            4.769E-03 9.920E-01 4.26E-03;
            5.166E-03 9.930E-01 5.46E-03;
            5.636E-03 9.930E-01 6.96E-03;
            6.199E-03 9.910E-01 8.76E-03;
            6.525E-03 9.890E-01 9.87E-03;
            6.888E-03 9.890E-01 1.11E-02;
            7.293E-03 9.880E-01 1.37E-02;
            7.749E-03 9.880E-01 1.60E-02;
            8.266E-03 9.890E-01 1.73E-02;
            8.551E-03 9.880E-01 1.81E-02;
            8.856E-03 9.880E-01 1.91E-02;
            8.984E-03 9.880E-01 1.97E-02;
            9.116E-03 9.870E-01 2.04E-02;
            9.253E-03 9.870E-01 2.13E-02;
            9.393E-03 9.860E-01 2.26E-02;
            9.537E-03 9.860E-01 2.44E-02;
            9.686E-03 9.870E-01 2.61E-02;
            9.840E-03 9.880E-01 2.77E-02;
            9.999E-03 9.890E-01 2.88E-02;
            1.016E-02 9.910E-01 2.92E-02;
            1.033E-02 9.920E-01 2.94E-02;
            1.042E-02 9.920E-01 2.97E-02;
            1.051E-02 9.920E-01 3.00E-02;
            1.060E-02 9.930E-01 3.03E-02;
            1.069E-02 9.930E-01 3.01E-02;
            1.078E-02 9.940E-01 3.04E-02;
            1.088E-02 9.940E-01 3.06E-02;
            1.097E-02 9.950E-01 3.07E-02;
            1.107E-02 9.950E-01 3.09E-02;
            1.117E-02 9.950E-01 3.09E-02;
            1.127E-02 9.960E-01 3.07E-02;
            1.137E-02 9.970E-01 3.05E-02;
            1.148E-02 9.970E-01 2.99E-02;
            1.159E-02 9.970E-01 2.93E-02;
            1.170E-02 9.960E-01 2.86E-02;
            1.181E-02 9.960E-01 2.81E-02;
            1.192E-02 9.940E-01 2.81E-02;
            1.204E-02 9.920E-01 2.86E-02;
            1.216E-02 9.920E-01 2.99E-02;
            1.228E-02 9.890E-01 3.24E-02;
            1.240E-02 9.890E-01 3.63E-02;
            1.252E-02 9.910E-01 3.96E-02;
            1.265E-02 9.940E-01 4.27E-02;
            1.278E-02 9.980E-01 4.42E-02;
            1.292E-02 1.001E+00 4.42E-02;
            1.305E-02 1.005E+00 4.39E-02;
            1.319E-02 1.007E+00 4.26E-02;
            1.333E-02 1.009E+00 4.11E-02;
            1.348E-02 1.011E+00 3.98E-02;
            1.362E-02 1.012E+00 3.86E-02;
            1.378E-02 1.013E+00 3.66E-02;
            1.393E-02 1.014E+00 3.51E-02;
            1.409E-02 1.014E+00 3.35E-02;
            1.425E-02 1.014E+00 3.21E-02;
            1.442E-02 1.014E+00 3.15E-02;
            1.459E-02 1.014E+00 3.12E-02;
            1.476E-02 1.016E+00 3.06E-02;
            1.494E-02 1.017E+00 2.89E-02;
            1.512E-02 1.017E+00 2.69E-02;
            1.531E-02 1.016E+00 2.50E-02;
            1.550E-02 1.016E+00 2.43E-02;
            1.569E-02 1.015E+00 2.38E-02;
            1.590E-02 1.014E+00 2.38E-02;
            1.610E-02 1.015E+00 2.45E-02;
            1.631E-02 1.016E+00 2.40E-02;
            1.653E-02 1.018E+00 2.40E-02;
            1.675E-02 1.021E+00 2.36E-02;
            1.687E-02 1.025E+00 2.40E-02;
            1.691E-02 1.028E+00 2.38E-02;
            1.696E-02 1.030E+00 1.96E-02;
            1.701E-02 1.036E+00 1.74E-02;
            1.705E-02 1.031E+00 6.06E-03;
            1.710E-02 1.026E+00 4.79E-03;
            1.722E-02 1.021E+00 4.61E-03;
            1.746E-02 1.015E+00 4.50E-03;
            1.771E-02 1.012E+00 4.45E-03;
            1.797E-02 1.009E+00 4.40E-03;
            1.851E-02 1.005E+00 4.33E-03;
            1.907E-02 1.001E+00 4.27E-03;
            2.066E-02 9.920E-01 4.51E-03;
            2.480E-02 9.740E-01 5.84E-03;
            3.100E-02 9.440E-01 8.64E-03;
            4.133E-02 8.820E-01 1.25E-02;
            6.199E-02 6.730E-01 2.67E-02;
            6.888E-02 5.640E-01 3.55E-02;
            7.749E-02 3.460E-01 6.48E-02;
            7.999E-02 2.520E-01 9.97E-02;
            8.266E-02 1.520E-01 1.87E-01;
            8.551E-02 1.070E-01 3.01E-01;
            8.856E-02 8.400E-02 4.04E-01;
            9.537E-02 6.500E-02 5.93E-01;
            1.033E-01 6.100E-02 7.67E-01;
            1.127E-01 6.100E-02 9.46E-01;
            1.378E-01 7.300E-02 1.35E+00;
        0.15 0.095390828 1.283666394;
        0.155 0.095510386 1.337393822;
        0.16 0.09903925 1.402928641;
        0.165 0.098692838 1.46616516;
        0.17 0.100850207 1.532569987;
        0.175 0.1069563 1.596539899;
        0.18 0.099715746 1.657661977;
        0.185 0.108316112 1.734368006;
        0.19 0.106567769 1.79116071;
        0.195 0.111513266 1.853411775;
        0.2 0.110803374 1.908606137;
        0.205 0.111587326 1.969936987;
        0.21 0.11365555 2.028057589;
        0.215 0.115928445 2.091850713;
        0.22 0.116173424 2.151998203;
        0.225 0.119771906 2.213039835;
        0.23 0.124315543 2.274896559;
        0.235 0.129276519 2.336773934;
        0.24 0.133002743 2.395514502;
        0.245 0.139688588 2.457358928;
        0.25 0.141162655 2.515219055;
        0.255 0.148765766 2.573777623;
        0.26 0.150722638 2.633214255;
        0.265 0.161465056 2.693431077;
        0.27 0.164610587 2.750035753;
        0.275 0.172365826 2.81099804;
        0.28 0.178635303 2.868973784;
        0.285 0.18587018 2.923276881;
        0.29 0.188953314 2.98248275;
        0.295 0.197315218 3.042340102;
        0.3 0.204991638 3.100858199;
        0.305 0.210097266 3.157347124;
        0.31 0.218816718 3.214790935;
        0.315 0.224454799 3.267473023;
        0.32 0.237666454 3.323775766;
        0.325 0.24464928 3.37932734;
        0.33 0.251892832 3.436779015;
        0.335 0.259391824 3.493749292;
        0.34 0.267481852 3.550147397;
        0.345 0.275201252 3.607178361;
        0.35 0.28349792 3.663518946;
        0.355 0.291774119 3.719829584;
        0.36 0.300125667 3.776251503;
        0.365 0.308578012 3.831608068;
        0.37 0.317597538 3.889198914;
        0.375 0.32692637 3.945995546;
        0.38 0.335956002 4.002607455;
        0.385 0.345714203 4.058265431;
        0.39 0.354901676 4.114238107;
        0.395 0.364968364 4.169676475;
        0.4 0.375150842 4.226433266;
        0.405 0.385211589 4.281283449;
        0.41 0.396086448 4.336805792;
        0.415 0.40706511 4.391435073;
        0.42 0.417647849 4.447407079;
        0.425 0.429543735 4.503499508;
        0.43 0.440996226 4.559109307;
        0.435 0.452837879 4.613682059;
        0.44 0.464232752 4.669101846;
        0.445 0.477070026 4.724044433;
        0.45 0.489220122 4.778319404;
        0.455 0.501228231 4.832828338;
        0.46 0.514817248 4.887948117;
        0.465 0.528042125 4.943260916;
        0.47 0.53987657 4.997345967;
        0.475 0.554932886 5.051331558;
        0.48 0.568005038 5.105940631;
        0.485 0.582771419 5.159226931;
        0.49 0.596705366 5.212874187;
        0.495 0.610784373 5.265933683;
        0.5 0.625686295 5.320477736;
        0.505 0.640306464 5.374848125;
        0.51 0.655709839 5.428163169;
        0.515 0.672565753 5.481545831;
        0.52 0.688336416 5.535955836;
        0.525 0.704045108 5.58799423;
        0.53 0.720793584 5.641767754;
        0.535 0.737603948 5.693773756;
        0.54 0.75446839 5.746496546;
        0.545 0.772330366 5.799008035;
        0.55 0.789405353 5.851936501;
        0.555 0.808351698 5.905288517;
        0.56 0.829205097 5.958236408;
        0.565 0.848630853 6.009090973;
        0.57 0.867376853 6.060226283;
        0.575 0.887661988 6.112687118;
        0.58 0.908569739 6.165184423;
        0.585 0.928308533 6.215605902;
        0.59 0.948955518 6.267417058;
        0.595 0.9714896 6.317103137;
        0.6 0.992465612 6.368986418;
        0.605 1.016073317 6.417351867;
        0.61 1.038145667 6.46681888;
        0.615 1.062059906 6.517052343;
        0.62 1.088160063 6.566589093;
        0.625 1.112663572 6.61475776;
        0.63 1.136328574 6.66312559;
        0.635 1.165731637 6.710806975;
        0.64 1.190265203 6.759417967;
        0.645 1.218505245 6.807110623;
        0.65 1.246364405 6.852329839;
        0.655 1.275302761 6.898361917;
        0.66 1.304382818 6.942363757;
        0.665 1.333854457 6.987643931;
        0.67 1.365410391 7.031951759;
        0.675 1.395892303 7.071104251;
        0.68 1.426024482 7.116287362;
        0.685 1.457391234 7.157652493;
        0.69 1.493230683 7.197481356;
        0.695 1.5274935 7.23603734;
        0.7 1.559751729 7.27391404;
        0.705 1.596346402 7.311383032;
        0.71 1.631621525 7.34575992;
        0.715 1.669698976 7.380372258;
        0.72 1.706780893 7.414249862;
        0.725 1.745260386 7.446739811;
        0.73 1.786398844 7.475374294;
        0.735 1.827772224 7.502849546;
        0.74 1.872189153 7.528352557;
        0.745 1.916802427 7.55216072;
        0.75 1.958355454 7.571403838;
        0.755 2.005349601 7.585211073;
        0.76 2.054224115 7.599462923;
        0.765 2.09984451 7.605086914;
        0.77 2.144528834 7.61216158;
        0.775 2.189915668 7.610567038;
        0.78 2.231500036 7.603853787;
        0.785 2.275479945 7.589387919;
        0.79 2.314643646 7.572223927;
        0.795 2.350258044 7.547894599;
        0.8 2.373653298 7.522581337;
        0.805 2.399438967 7.486541749;
        0.81 2.409296598 7.449477169;
        0.815 2.414395973 7.410756231;
        0.82 2.410049326 7.369351244;
        0.825 2.399462094 7.325600203;
        0.83 2.375277499 7.288411569;
        0.835 2.342102357 7.253098886;
        0.84 2.301774531 7.225729444;
        0.845 2.257656113 7.204374444;
        0.85 2.204898553 7.188085811;
        0.855 2.148111401 7.177700369;
        0.86 2.092718783 7.181328004;
        0.865 2.03127715 7.184642431;
        0.87 1.973568962 7.198887028;
        0.875 1.915492386 7.218248477;
        0.88 1.854996083 7.243411521;
        0.885 1.802564727 7.274079983;
        0.89 1.750120449 7.309782595;
        0.895 1.694671727 7.350096072;
        0.9 1.64715608 7.393045594;
        0.905 1.602399525 7.439468008;
        0.91 1.557188172 7.48978204;
        0.915 1.516837858 7.542665118;
        0.92 1.483545494 7.592621895;
        0.925 1.447230511 7.647710517;
        0.93 1.415213787 7.70342976;
        0.935 1.385433286 7.756057335;
        0.94 1.355589911 7.812465936;
        0.945 1.325884954 7.872100794;
        0.95 1.302164463 7.928721983;
        0.955 1.280288795 7.985307721;
        0.96 1.25976713 8.04593147;
        0.965 1.23821822 8.1051949;
        0.97 1.218188621 8.163760378;
        0.975 1.201067407 8.22268587;
        0.98 1.182294465 8.279240316;
        0.985 1.167631515 8.338247019;
        0.99 1.153696234 8.399524689;
        0.995 1.140905235 8.457757667;
        1 1.126639087 8.511598888;
        1.005 1.114680776 8.571960874;
        1.01 1.102964941 8.629920722;
        1.015 1.09358083 8.687515124;
        1.02 1.08727974 8.74843856;
        1.025 1.081445553 8.809967337;
        1.03 1.071397488 8.867857271;
        1.035 1.061537336 8.920081923;
        1.04 1.055494597 8.978255334;
        1.045 1.051806876 9.036830069;
        1.05 1.043454926 9.083765639;
        1.055 1.036482441 9.148731926;
        1.06 1.033133204 9.204040646;
        1.065 1.02747793 9.264432573;
        1.07 1.025591906 9.318521275;
        1.075 1.024418445 9.373122913;
        1.08 1.020244607 9.42862127;
        1.085 1.015304156 9.479917855;
        1.09 1.014378517 9.5368739;
        1.095 1.011890304 9.593431189;
        1.1 1.009905005 9.649917519;
        1.105 1.008451539 9.704482186;
        1.11 1.005344855 9.750882949;
        1.115 1.00382405 9.806001363;
        1.12 1.004294808 9.860152537;
        1.125 1.003882164 9.917498646;
        1.13 1.003972742 9.968398129;
        1.135 1.00500573 10.02382345;
        1.14 1.002828586 10.07500199;
        1.145 1.003941179 10.12708734;
        1.15 1.005317282 10.18335224;
        1.155 1.003754384 10.23706613;
        1.16 1.008818757 10.28628773;
        1.165 1.007932489 10.34146643;
        1.17 1.008357257 10.39215013;
        1.175 1.012688334 10.44200832;
        1.18 1.015213183 10.49336256;
        1.185 1.014759181 10.5517646;
        1.19 1.017683803 10.60545522;
        1.195 1.01813905 10.65817091;
        1.2 1.021736807 10.70579031;
        1.205 1.022859293 10.75807795;
        1.21 1.024257787 10.80782605;
        1.215 1.026450838 10.85926339;
        1.22 1.029738722 10.9110202;
        1.225 1.035047848 10.96181208;
        1.23 1.038667581 11.01266681;
        1.235 1.039978895 11.06849385;
        1.24 1.043389606 11.11569439;
        1.245 1.046657092 11.16984741;
        1.25 1.049913391 11.21616325;
        1.255 1.051297547 11.26655345;
        1.26 1.056495057 11.31619114;
        1.265 1.057188404 11.36079431;
        1.27 1.061492702 11.41081797;
        1.275 1.066383574 11.47288865;
        1.28 1.070500901 11.51937378;
        1.285 1.073080696 11.56250415;
        1.29 1.078091053 11.61775711;
        1.295 1.081263285 11.66829595;
        1.3 1.08667656 11.71967858;
        1.305 1.090905752 11.7737027;
        1.31 1.094227716 11.82020872;
        1.315 1.097106966 11.86438552;
        1.32 1.104427092 11.91930196;
        1.325 1.108138114 11.96737106;
        1.33 1.109282178 12.01542787;
        1.335 1.112492808 12.06389822;
        1.34 1.123987022 12.1162431;
        1.345 1.129233973 12.16532652;
        1.35 1.132981867 12.21775953;
        1.355 1.135566133 12.26700903;
        1.43 1.20897633 12.99198306;
        1.435 1.215774284 13.0402495;
        1.44 1.222359347 13.08984959;
        1.445 1.225637477 13.13438949;
        1.45 1.232348412 13.19350911;
        1.455 1.235053048 13.23424936;
        1.46 1.23989324 13.2825952;
        1.465 1.245674136 13.32973008;
        1.47 1.25590184 13.37524914;
        1.475 1.252421713 13.42119816;
        1.48 1.265153644 13.46516297;
        1.485 1.264792932 13.51960433;
        1.49 1.275023692 13.56523813;
        1.495 1.277693673 13.61515704;
        1.5 1.285699173 13.66027168;
        1.505 1.290149697 13.70964938;
        1.51 1.296809948 13.75760575;
        1.515 1.304059962 13.81378197;
        1.52 1.313924676 13.8577198;
        1.525 1.31627889 13.90548777;
        1.53 1.325289614 13.94978109;
        1.535 1.332565366 13.9955611;
        1.54 1.338103685 14.0463704;
        1.545 1.342317392 14.08906725;
        1.55 1.347399401 14.13278052;
        1.555 1.354287918 14.18605286;
        1.56 1.36162657 14.2300396;
        1.565 1.367953829 14.27204602;
        1.57 1.374881699 14.33214219;
        1.575 1.374949427 14.37325593;
        1.58 1.385662006 14.41873986;
        1.585 1.390425967 14.47146449;
        1.59 1.400170689 14.51965833;
        1.595 1.39786136 14.56653756;
        1.6 1.400984694 14.60865354;
        1.605 1.402928097 14.65292487;
        1.61 1.421414675 14.70647543;
        1.615 1.424742484 14.75635087;
        1.62 1.436002763 14.82100212;
        1.625 1.439358574 14.84862799;
        1.63 1.44609075 14.8956765;
        1.635 1.447904247 14.94712102;
        1.64 1.45643876 14.99170484;
        1.645 1.458873135 15.02392461;
        1.65 1.469917493 15.08179888;
        1.655 1.478177828 15.1223348;
        1.66 1.474977808 15.17582154;
        1.665 1.477843053 15.23233469;
        1.67 1.462292742 15.2773787;
        1.675 1.425778483 15.34740513;
        1.68 1.427335087 15.38951869;
        1.685 1.454908413 15.4724516;
        1.69 1.514176047 15.49621661;
        1.695 1.555668449 15.53351552;
        1.7 1.584018511 15.55632073;
        2.00 2.1962737 20.969371;
        2.11 2.4038118 22.062867;
        2.22 2.6179915 23.257800;
        2.35 2.8842412 24.591905;
        2.50 3.1815630 26.073820;
        2.67 3.5693590 27.735962;
        2.86 4.0097886 29.575977;
        3.08 4.5104289 31.653186;
        3.33 5.0466806 34.058286;
        3.64 5.7888000 36.878956;
        4.00 6.5579910 40.184716;
        4.44 7.6669569 44.210650;
        5.00 8.8866481 49.131664;
        5.71 10.625810 55.544522;
        6.67 13.385937 63.967532;
        8.00 17.708575 75.320562;
        10.0 25.832564 90.720430;
        11.1 30.167884 98.172167;
        12.5 35.492290 107.26425;
        14.3 42.226361 117.86671;
        16.7 50.744867 131.07397;
        20.0 61.871514 148.17314;
        22.2 68.262241 159.23227;
        25.0 76.681189 174.09538;
        28.6 88.918176 192.23917;
        33.3 108.95823 213.37170;
        40.0 134.05166 235.70537;
        44.4 149.32871 248.55977;
        50.0 167.07810 263.04366;
        57.1 188.04844 279.77106;
        66.7 213.38844 299.72607;
        80.0 245.00980 324.59867;
        100 286.08777 357.47647;
        125 329.97183 393.77470;
        154 374.73526 431.44791;
        200 436.98909 485.19932
        ])
    dat[material]
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

end