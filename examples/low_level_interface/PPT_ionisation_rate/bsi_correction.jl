# Barrier-suppression (over-the-barrier) corrections to the PPT ionisation rate.
#
# Two-panel comparison for He @ 800 nm:
#   (a) the multiplicative correction factor W_corr / W_PPT vs E / E_b
#   (b) the resulting rate vs field for bare PPT, PPT × Tong–Lin and PPT × Zhang.
#
# Tong & Lin, J. Phys. B 38, 2593 (2005); Zhang, Lan & Lu, Phys. Rev. A 90, 043410 (2014).
# See the CORRECTNESS NOTE in src/Ionisation.jl on the sign typo in Zhang eq. (8): if the
# Zhang factor in panel (a) ever rises above 1, that typo has been reintroduced.

import Luna: Ionisation, PhysData
import Luna.PhysData: au_Efield, c, ε_0
import PyPlot: plt, pygui

pygui(true)

λ0 = 800e-9
gas = :He
Ip_au = PhysData.ionisation_potential(gas) / PhysData.au_energy
Z = 1.0
Eb_au = Ip_au^2 / (4Z)
Eb = Eb_au * au_Efield                      # V/m

none = Ionisation.IonRatePPT(gas, λ0; bsi=:none)
tl   = Ionisation.IonRatePPT(gas, λ0; bsi=:tonglin)
zh   = Ionisation.IonRatePPT(gas, λ0; bsi=:zhang)

x  = range(0.3, 4.7; length=400)            # E / E_b
E  = collect(x) .* Eb
Wn = none.(E); Wt = tl.(E); Wz = zh.(E)
intensity(Ef) = 0.5c*ε_0*Ef^2 / 1e4         # W/cm^2

fig, (axL, axR) = plt.subplots(1, 2; figsize=(13, 5))

# (a) correction factor
axL.axvspan(0.5, 1.3; color="#ffe08a", alpha=0.55)        # operating range
axL.plot(x, Wt ./ Wn; lw=2.5, color="#c0392b", label="Tong–Lin (2005), α=7")
axL.plot(x, Wz ./ Wn; lw=2.5, color="#1f5fa6", label="Zhang et al. (2014)")
axL.axvline(2.0; ls=":", color="#c0392b"); axL.axvline(4.5; ls=":", color="#1f5fa6")
axL.set_xlabel("E / E_b"); axL.set_ylabel("W_corr / W_PPT")
axL.set_ylim(0, 1.02); axL.legend(frameon=false); axL.grid(alpha=0.25)
axL.set_title("(a) correction factor")

# (b) resulting rate
axR.semilogy(x, Wn; ls="--", color="0.55", lw=2, label="bare PPT (overestimates)")
axR.semilogy(x, Wt; color="#c0392b", lw=2.5, label="PPT × Tong–Lin")
axR.semilogy(x, Wz; color="#1f5fa6", lw=2.5, label="PPT × Zhang")
axR.axvspan(0.5, 1.3; color="#ffe08a", alpha=0.55)
axR.set_xlabel("E / E_b"); axR.set_ylabel("rate (1/s)")
axR.legend(frameon=false); axR.grid(alpha=0.25, which="both")
axR.set_title("(b) resulting He rate")

fig.suptitle("He barrier-suppression corrections  (E_b = $(round(Eb_au; digits=3)) a.u. " *
             "= $(round(Eb/1e11; digits=2))×10¹¹ V/m)")
fig.tight_layout()
#fig.savefig(joinpath(dirname(@__FILE__), "bsi_correction_He.png"); dpi=150)
