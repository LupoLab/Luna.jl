# N = 5 soliton

using Luna

γ = 0.1
β2 = -1e-26
N = 4.0
τ0 = 280e-15
τfwhm = (2*log(1 + sqrt(2)))*τ0
fr = 0.18
P0 = N^2*abs(β2)/((1-fr)*γ*τ0^2)
flength = π*τ0^2/abs(β2)
βs =  [0.0, 0.0, β2]

λ0 = 835e-9
λlims = [450e-9, 8000e-9]
trange = 4e-12

output = prop_gnlse(γ, flength, βs; λ0, τfwhm, power=P0, pulseshape=:sech, λlims, trange,
                    raman=false, shock=false, fr, shotnoise=false)

##

Plotting.prop_2D(output, :ω, dBmin=-100.0,  λrange=(720e-9,1000e-9), trange=(-300e-15, 300e-15), oversampling=1)
