using Luna
import Luna.PhysData: wlfreq
import FFTW
import Hankel

# Deep-UV Gaussian beam in SiO₂ — envelope, radially symmetric
λ0 = 240e-9
τfwhm = 2e-15   # broad bandwidth across TPA region
energy = 150e-9
w0 = 15e-6        # beam waist
L = 10e-6        # 100 µm propagation (thin slab)
R = 250e-6        # Hankel aperture
N = 256            # radial points

material = :MgF2

grid = Grid.EnvGrid(L, λ0, (160e-9, 600e-9), 1e-12)
q = Hankel.QDHT(R, N, dim=2)

densityfun(z) = 1.0  # solid — χ³ already bulk value
nfun = PhysData.ref_index_fun(material, lookup=false)
normfun = NonlinearRHS.const_norm_radial(grid, q, nfun)
linop = LinearOps.make_const_linop(grid, q, nfun)

χ3 = PhysData.χ3(material)
β₂_ω = PhysData.β₂_TPA.(grid.ω, material)
tpa = Nonlinear.TPAResponse(grid.ω, β₂_ω)

inputs = Fields.GaussGaussField(λ0=λ0, τfwhm=τfwhm, energy=energy, w0=w0)

# With TPA
responses_tpa = (Nonlinear.Kerr_env(χ3), tpa)
Eω_tpa, transform_tpa, FT_tpa = Luna.setup(grid, q, densityfun, normfun,
responses_tpa, inputs)
output_tpa = Output.MemoryOutput(0, grid.zmax, 201)
Luna.run(Eω_tpa, grid, linop, transform_tpa, FT_tpa, output_tpa)

# Without TPA
responses_notpa = (Nonlinear.Kerr_env(χ3),)
Eω_notpa, transform_notpa, FT_notpa = Luna.setup(grid, q, densityfun, normfun,
responses_notpa, inputs)
output_notpa = Output.MemoryOutput(0, grid.zmax, 201)
Luna.run(Eω_notpa, grid, linop, transform_notpa, FT_notpa, output_notpa)

# Energy comparison (sum over all radial k-modes)
Eωk_tpa_in = output_tpa.data["Eω"][:, :, 1]
Eωk_tpa_out = output_tpa.data["Eω"][:, :, end]
Eωk_notpa_out = output_notpa.data["Eω"][:, :, end]

energy_in = sum(abs2, Eωk_tpa_in)
energy_out_tpa = sum(abs2, Eωk_tpa_out)
energy_out_notpa = sum(abs2, Eωk_notpa_out)

# With TPA, output energy should be less than without
@info "Energy comparison: TPA output = $energy_out_tpa, No TPA output = $energy_out_notpa"

# Spectral suppression: TPA stronger at shorter wavelengths (higher ω)
# Sum over radial modes for spectral comparison
spec_tpa = dropdims(sum(abs2, Eωk_tpa_out, dims=2), dims=2)
spec_notpa = dropdims(sum(abs2, Eωk_notpa_out, dims=2), dims=2)

ω_below = 2π * PhysData.c / 300e-9   # below SiO₂ TPA edge (~299 nm)
ω_above = 2π * PhysData.c / 220e-9   # well above TPA edge

idx_below = argmin(abs.(grid.ω .- ω_below))
idx_above = argmin(abs.(grid.ω .- ω_above))

ratio_below = spec_tpa[idx_below] / spec_notpa[idx_below]
ratio_above = spec_tpa[idx_above] / spec_notpa[idx_above]
# Above-edge suppression should be stronger (lower ratio)
@info "Spectral suppression: ratio above edge = $ratio_above, ratio below edge = $ratio_below"

λ, Iλtpa = Processing.getIω(output_tpa, :λ, L)
λ, Iλnotpa = Processing.getIω(output_notpa, :λ, L)

using PyPlot
figure()
plot(λ*1e9, Iλtpa[:,1,1], label="With TPA")
plot(λ*1e9, Iλnotpa[:,1,1], label="Without TPA")
xlabel("Wavelength (nm)")
ylabel("On-axis spectral Intensity (a.u.)")
legend()
title("Spectral suppression from TPA in SiO₂")
xlim(180, 400)
