using Luna
import DelimitedFiles: readdlm
import PyPlot: plt, pygui
import Printf: @sprintf
import GSL: hypergeom
import SpecialFunctions: dawson, gamma
import Luna.Modes: hquadrature

this_folder = dirname(@__FILE__)
## Ilkov et al
k = collect(range(10, stop=12, length=50))
E = 10 .^ k
# ppt = Ionisation.ionrate_PPT(:He, 10.6e-6, E)
ppt_cycavg = Ionisation.ionrate_PPT(:He, 10.6e-6, E, cycle_average=true, sum_tol=1e-6, stark_shift=false)
ppt_cycavg_rough = Ionisation.ionrate_PPT(:He, 10.6e-6, E, cycle_average=true, sum_tol=1e-4, stark_shift=false)
ppt_cycavg_intg = Ionisation.ionrate_PPT(:He, 10.6e-6, E, cycle_average=true, sum_integral=true, stark_shift=false)
adk = Ionisation.ionrate_ADK(:He, E)

dat = readdlm(joinpath(this_folder, "Ilkov_PPT_He.csv"), ',') # ionrate [1/s] vs field [V/cm]

fig = plt.figure()
# plt.loglog(E, ppt, label="PPT")
plt.loglog(E, ppt_cycavg/2, label="PPT cycle averaged")
plt.loglog(E, ppt_cycavg_rough/2, label="PPT cycle averaged, rough tolerance")
plt.loglog(E, ppt_cycavg_intg/2, label="PPT cycle averaged, integral for sum")
# plt.loglog(E, adk, label="ADK")
plt.loglog(dat[:, 1].*100, dat[:, 2], label="Ilkov et al. PPT")
plt.xlim(1e10, 1e12)
plt.ylim(0.1, 1e18)
plt.legend()
plt.xlabel("Field strength (V/m)")
plt.ylabel("Ionisation rate (s⁻¹)")
plt.title("He ionisation at 10.6 μm (Stark shift off, divided by 2)")


## Chang, Fundamentals of Attosecond Optics
dat = readdlm(joinpath(this_folder, "Chang_PPT.csv"), ',') # ionrate [1/fs] vs intensity [1e14 W/cm^2]
intensity = range(0.1, 25; length=100) * 1e18 # W/m^2
E = Tools.intensity_to_field.(intensity)
ppt = Ionisation.ionrate_PPT(:He, 390e-9, E)
ppt_cycavg = Ionisation.ionrate_PPT(:He, 390e-9, E, cycle_average=true)
adk = Ionisation.ionrate_ADK(:He, E)

s = sortperm(dat[:, 1])

plt.figure()
plt.loglog(intensity*1e-18, ppt*1e-15/2, label="PPT")
plt.loglog(intensity*1e-18, adk*1e-15, label="ADK")
plt.loglog(intensity*1e-18, ppt_cycavg*1e-15/2, label="PPT cycle averaged")
plt.loglog(dat[s, 1], dat[s, 2], label="Chang PPT")
plt.ylim(1e-14, 1)
plt.xlim(extrema(intensity.*1e-18))
plt.legend()
plt.xlabel("Intensity (10\$^{14}\$ W/cm\$^2\$)")
plt.ylabel("Ionisation rate (1/fs)")
plt.title("PPT rate divided by 2")

## Couairon
Ip = 12.063 * PhysData.electron
dat = readdlm(joinpath(this_folder, "Couairon_PPT.csv"), ',') # ionrate [1/s] vs intensity [W/cm^2]
k = collect(range(12, 15, length=100))
intensity = 10 .^ k .* 1e4 # W/m^2
E = Tools.intensity_to_field.(intensity)
ppt = Ionisation.ionrate_PPT(Ip, 800e-9, 1, 0, E)
ppt_cycavg = Ionisation.ionrate_PPT(Ip, 800e-9, 1, 0, E; cycle_average=true)
adk = Ionisation.ionrate_ADK(Ip, E)

s = sortperm(dat[:, 1])

plt.figure()
# plt.loglog(intensity*1e-4, ppt, label="PPT")
plt.loglog(intensity*1e-4, adk, label="ADK")
plt.loglog(intensity*1e-4, ppt_cycavg/2, label="PPT cycle averaged")
plt.loglog(dat[s, 1], dat[s, 2], "--", label="Couairon PPT")
plt.ylim(1, 1e17)
plt.xlim(extrema(intensity.*1e-4))
plt.legend()
plt.xlabel("Intensity (W/cm\$^2\$)")
plt.ylabel("Ionisation rate (1/s)")
plt.title("O\$_2\$ ionisation at 800 nm (PPT divided by 2)")

## Gonzales et al
dat = readdlm(joinpath(this_folder, "Gonzalez_PPT_Ar.csv"), ',') # ionrate [1/s] vs intensity [W/cm^2]
k = collect(range(12, 16, length=100))
intensity = 10 .^ k .* 1e4 # W/m^2
E = Tools.intensity_to_field.(intensity)
λ0 = 800e-9
gas = :Ar
# ppt = Ionisation.ionrate_PPT(gas, λ0, E)
sum_tol = 1e-5
ppt_cycavg = Ionisation.ionrate_PPT(gas, λ0, E; cycle_average=true, sum_tol)
ppt_cycavg_m0 = Ionisation.ionrate_PPT(gas, λ0, E; cycle_average=true, sum_tol, msum=false)
ppt_cycavg_msum_nostark = Ionisation.ionrate_PPT(gas, λ0, E; cycle_average=true, stark_shift=false)

s = sortperm(dat[:, 1])

plt.figure()
# plt.loglog(intensity*1e-4, ppt, label="PPT")
plt.loglog(intensity*1e-4, ppt_cycavg/2, label="PPT cycle averaged, sum over m")
plt.loglog(intensity*1e-4, ppt_cycavg_m0/2, ":", label="PPT cycle averaged, m=0")
plt.loglog(intensity*1e-4, ppt_cycavg_msum_nostark/2, "--", label="PPT cycle averaged, sum over m, no Stark shift")
plt.loglog(dat[s, 1], dat[s, 2], "--", label="Gonzalez PPT")
plt.ylim(1, 1e18)
plt.xlim(extrema(intensity.*1e-4))
plt.legend()
plt.xlabel("Intensity (W/cm\$^2\$)")
plt.ylabel("Ionisation rate (1/s)")
plt.title("Ar ionisation at 800 nm (PPT divided by 2)")

## Sum convergence test
k = collect(range(12, 16, length=100))
intensity = 10 .^ k .* 1e4 # W/m^2
E = Tools.intensity_to_field.(intensity)
sum_tol = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
ppt = zeros((length(E), length(sum_tol)))

gas = :He
λ0 = 1800e-9
for (idx, sti) in enumerate(sum_tol)
    ppt[:, idx] .= Ionisation.ionrate_PPT(gas, λ0, E; sum_tol=sti)
end
##
cols = plt.get_cmap().(collect(range(0, 0.8, length(sum_tol))))
fig = plt.figure()
for idx in eachindex(sum_tol)
    plt.loglog(intensity*1e-4, ppt[:, idx];
            c=cols[idx], alpha=0.8, label=@sprintf("%.0e", sum_tol[idx]))
end
# plt.xlim(5e14, 5e15)
# plt.ylim(1e11, 1e16)
plt.xlabel("Intensity (W/cm²)")
plt.ylabel("Ionisation rate (s⁻¹)")
plt.legend(;title="Sum convergence tolerance")
plt.title(@sprintf("%s, %.0f nm", gas, λ0*1e9))

## Comparing ways of calculating φ
function φ(m, x)
    if m == 0
        return dawson(x)
    end
    mabs = abs(m)
    return (exp(-x^2)
        * sqrt(π)
        * x^(2mabs+1)
        * gamma(mabs+1)
        * hypergeom(1/2, 3/2 + mabs, x^2)
        / (2*gamma(3/2 + mabs)))
end

function φhard(m, x)
    i, _ = hquadrature(0, x) do y
        y = BigFloat(y)
        x = BigFloat(x)
        (x^2 - y^2)^(abs(m))*exp(y^2)
    end
    return exp(-x^2) * i
end

m = [0, 1, 2]
x = collect(range(0, 26, 512))

φ1 = mapreduce(hcat, m) do mi
    φ.(mi, x)
end
φ2 = Float64.(
    mapreduce(hcat, m) do mi
        Ionisation.φ.(mi, 2x)
    end
)

plt.figure()
for (idx, mi) in enumerate(m)
    plt.plot(x, φ1[:, idx])
    plt.plot(2x, φ2[:, idx], linestyle="--")
end