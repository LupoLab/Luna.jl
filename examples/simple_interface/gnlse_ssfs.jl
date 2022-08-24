using Luna

γ = 0.1
β2 = -1e-26
N = 1.0
τ0 = 10e-15
fr = 0.18
P0 = N^2*abs(β2)/((1 - fr)*γ*τ0^2)
flength = pi/2*τ0^2/abs(β2)*90
βs =  [0.0, 0.0, β2]

λ0 = 835e-9
λlims = [450e-9, 8000e-9]
trange = 12e-12

output = prop_gnlse(γ, flength, βs; λ0, τfwhm=1.763*τ0, power=P0, pulseshape=:sech, λlims, trange,
                    raman=true, shock=false, fr, shotnoise=false, ramanmodel=:sdo, τ1=12.2e-15, τ2=32e-15,
                    saveN=601)

##
Plotting.pygui(true)
Plotting.prop_2D(output, :ω, dBmin=-40.0,  λrange=(700e-9,1500e-9), trange=(-50e-15, 6000e-15), oversampling=1)
