module Ionisation
import SpecialFunctions: gamma
import Luna.PhysData: ħ, electron, m_e, au_energy, ionisation_potential

function ionrate_fun_ADK(ionpot::Float64)
    nstar = sqrt(0.5/(ionpot/au_energy))
    cn_sq = 2^(2*nstar)/(nstar*gamma(nstar+1)*gamma(nstar))
    ω_p = ionpot/ħ
    ω_t_prefac = electron/sqrt(2*m_e*ionpot)

    ionrate = let nstar=nstar, cn_sq=cn_sq, ω_p=ω_p, ω_t_prefac=ω_t_prefac
        E -> @. ω_p*cn_sq*(4*ω_p/(ω_t_prefac*E))^(2*nstar-1)*exp(-4/3*ω_p/(ω_t_prefac*E))
    end

    return ionrate    
end

function ionrate_fun_ADK(material::Symbol)
    return ionrate_fun_ADK(ionisation_potential(material))
end

function ionrate_ADK(IP_or_material, E)
    return ionrate_fun_ADK(IP_or_material)(E)
end
end