module Ionisation
import SpecialFunctions: gamma
import GSL: hypergeom
import Luna.PhysData: c, ħ, electron, m_e, au_energy, au_time, au_Efield
import Luna.PhysData: ionisation_potential, quantum_numbers
import Luna: Maths

function ionrate_fun!_ADK(ionpot::Float64)
    nstar = sqrt(0.5/(ionpot/au_energy))
    cn_sq = 2^(2*nstar)/(nstar*gamma(nstar+1)*gamma(nstar))
    ω_p = ionpot/ħ
    ω_t_prefac = electron/sqrt(2*m_e*ionpot)

    ionrate! = let nstar=nstar, cn_sq=cn_sq, ω_p=ω_p, ω_t_prefac=ω_t_prefac
        function ionrate!(out, E)
            @. out = ω_p*cn_sq*(4*ω_p/(ω_t_prefac*abs(E)))^(2*nstar-1)*exp(-4/3*ω_p/(ω_t_prefac*abs(E)))
            out[E .< 1e6] .= 0
        end
    end

    return ionrate!  
end

function ionrate_fun!_ADK(material::Symbol)
    return ionrate_fun!_ADK(ionisation_potential(material))
end

function ionrate_ADK(IP_or_material, E)
    out = zero(E)
    ionrate_fun!_ADK(IP_or_material)(out, E)
    return out
end

function ionrate_ADK(IP_or_material, E::Number)
    out = [zero(E)]
    ionrate_fun!_ADK(IP_or_material)(out, [E])
    return out[1]
end


function ionrate_fun_PPT(ionpot::Float64, λ0, Z, l; sum_tol=1e-4)
    Ip_au = ionpot / au_energy
    ns = Z/sqrt(2*Ip_au)
    ls = ns-1
    Cnl2 = 2^(2*ns)/(ns*gamma(ns + ls + 1)*gamma(ns - ls))

    ω0 = 2π*c/λ0
    ω0_au = au_time*ω0
    E0_au = (2*Ip_au)^(3/2)

    ionrate = let ω0_au=ω0_au, Cnl2=Cnl2, ns=ns, sum_tol=sum_tol
        function ionrate(E)
            E_au = @. abs(E)/au_Efield
            g = ω0_au/sqrt(2*Ip_au)/E_au
            g2 = g*g
            β = 2*g/sqrt(1 + g2)
            α = 2*(asinh(g) - g/sqrt(1+g2))
            Up_au = E_au^2/(4*ω0_au^2)
            Uit_au = Ip_au + Up_au
            v = Uit_au/ω0_au
            G = 3/(2*g)*((1 + 1/(2*g2))*asinh(g) - sqrt(1 + g2)/(2*g))
            ret = 0
            divider = 0
            for m = -l:l
                divider += 1
                mabs = abs(m)
                flm = ((2*l + 1)*factorial(l + mabs)
                    / (2 ^ mabs*factorial(mabs)*factorial(l - mabs)))
                Am = 4/(sqrt(3*π)*factorial(mabs))*g2/(1 + g2)
                lret = sqrt(3/(2*π))*Cnl2*flm*Ip_au
                lret *= (2*E0_au/(E_au*sqrt(1 + g2))) ^ (2*ns - mabs - 3/2)
                lret *= Am*exp(-2*E0_au*G/(3*E_au))
                # lret *= sqrt(π*E0_au/(3*E_au))
                k = ceil(v)
                n0 = ceil(v)
                sumfunc = let k=k, β=β, m=m
                    function sumfunc(x, n)
                        diff = n-v
                        return x + exp(-α*diff)*φ(m, sqrt(β*diff))
                    end
                end
                s, success, steps = Maths.aitken_accelerate(sumfunc, 0, n0=n0, rtol=sum_tol)
                lret *= s
                ret += lret
            end
            return ret/(au_time*divider)
        end
    end
    return ionrate
end

function φ(m, x)
    mabs = abs(m)
    return (exp(-x^2)
            * sqrt(π)
            * x^(mabs+1)
            * gamma(mabs+1)
            * hypergeom(1/2, 3/2 + mabs, x^2)
            / (2*gamma(3/2 + mabs)))
end

function ionrate_fun_PPT(material::Symbol)
    return ionrate_fun_PPT(ionisation_potential(material))
end

function ionrate_PPT(ionpot, λ0, Z, l, E::AbstractArray)
    return ionrate_fun_PPT(ionpot, λ0, Z, l).(E)
end

function ionrate_PPT(ionpot, λ0, Z, l, E::Number)
    return ionrate_fun_PPT(ionpot, λ0, Z, l)(E)
end

function ionrate_PPT(material::Symbol, λ0, E)
    n, l, Z = quantum_numbers(material)
    ip = ionisation_potential(material)
    return ionrate_PPT(ip, λ0, Z, l, E)
end

end