# Post-processing & compression

After propagating a pulse through a fibre (or other nonlinear medium) with Luna, the
output frequency-domain field `Eω` can be further manipulated to simulate optical
elements encountered after the fibre: mirrors, bulk materials, gratings, prisms, and
more. This page describes how to use the `prop_*` (propagation) and `optcomp_*`
(optimised compression) functions in the [`Fields`](modules/Fields.md) module.

All functions follow a consistent convention:
- **In-place** versions are named `prop_X!(Eω, ω, ...)` and modify `Eω` directly.
- **Grid dispatch**: `prop_X!(Eω, grid::Grid.AbstractGrid, ...)` forwards to the `ω` version.
- **Copy** versions `prop_X(Eω, ...)` return a new array without modifying the input.
- **Units** are SI throughout: wavelengths in metres, angles in radians, time in
  seconds, etc.
- Spectral phase is applied as ``E_\omega \leftarrow E_\omega \cdot e^{-i\phi(\omega)}``.

## Extracting fields from simulation output

After running a simulation with `prop_capillary` (or the low-level interface), the
output field at the end of the fibre can be retrieved using [`Processing.getEω`](@ref):

```julia
using Luna

output = prop_capillary(a, flength, gas, pressure; λ0, τfwhm, energy, λlims, trange)
# Extract frequency-domain field at the fibre output (last z-step)
ω, Eω = Processing.getEω(output)
```

The temporal field can be obtained with [`Processing.getEt`](@ref).

## Taylor-expanded spectral phase

[`Fields.prop_taylor!`](@ref) / [`Fields.prop_taylor`](@ref) add spectral phase given
as Taylor expansion coefficients `ϕs = [ϕ₀, ϕ₁, ϕ₂, ϕ₃, ...]` around a central
wavelength `λ0`:

```math
\phi(\omega) = \sum_{n=0} \frac{\phi_n}{n!} (\omega - \omega_0)^n
```

where `ϕ₀` is the absolute phase (CEP), `ϕ₁` is the group delay, `ϕ₂` is the
group-delay dispersion (GDD), `ϕ₃` is the third-order dispersion (TOD), etc.

```julia
# Add 1000 fs² of GDD and 500 fs³ of TOD at 800 nm
λ0 = 800e-9
ϕs = [0, 0, 1000e-30, 500e-45]  # [CEP, group delay, GDD, TOD]
Eω_chirped = Fields.prop_taylor(Eω, grid, ϕs, λ0)
```

[`Fields.optcomp_taylor`](@ref) finds the Taylor coefficients that maximise the peak
power of the pulse (i.e. optimal compression):

```julia
ϕs_opt, Eω_compressed = Fields.optcomp_taylor(Eω, grid, λ0; order=3)
# ϕs_opt contains the optimal [ϕ₀, ϕ₁, ϕ₂, ϕ₃]
```

This is useful as a diagnostic to determine how much dispersion is present in a pulse,
and as a first step before choosing a physical compressor.

## Material propagation

[`Fields.prop_material!`](@ref) / [`Fields.prop_material`](@ref) linearly propagate the
field through a given `thickness` of a bulk material. The refractive index is obtained
from `PhysData.ref_index_fun`. Available materials include `:SiO2` (fused silica),
`:BK7`, `:CaF2`, `:MgF2`, `:BaF2`, `:KBr`, and gases like `:Air`.

```julia
# Propagate through 2 mm of fused silica
Eω_out = Fields.prop_material(Eω, grid, :SiO2, 2e-3, λ0)
```

When `λ0` is given, the overall group delay at that wavelength is subtracted so that
only the dispersive (chirp) part of the phase remains.

[`Fields.optcomp_material`](@ref) finds the material thickness that maximises peak power:

```julia
# Find optimal SiO2 thickness for compression (searching from -10 mm to +10 mm)
d_opt, Eω_compressed = Fields.optcomp_material(Eω, grid, :SiO2, λ0, -1e-2, 1e-2)
```

Negative thickness values correspond to removing material from the beam path (e.g.
adjusting a wedge pair). This is commonly used for fine-tuning compression with glass
wedges.

## Mirror reflection

[`Fields.prop_mirror!`](@ref) / [`Fields.prop_mirror`](@ref) apply the complex transfer
function of a mirror (reflectivity and spectral phase) for a given number of reflections.

Named mirror types are looked up from the `PhysData` database:

```julia
# 10 bounces on PC70 chirped mirrors
Eω_out = Fields.prop_mirror(Eω, grid, 10, :PC70)
```

Custom mirror data can also be provided:

```julia
Fields.prop_mirror!(Eω, grid, reflections, λR, R, λGDD, GDD, λ0, λmin, λmax)
```

where `λR, R` are the reflectivity curve and `λGDD, GDD` are the GDD per bounce.

## Grating compression

A four-grating (double-pass) compressor provides tuneable negative GDD, well-suited
for compressing broadband positively chirped pulses from SPM broadening.

### Helper functions

[`Fields.littrow_angle`](@ref) calculates the Littrow angle (angle of incidence where
the diffracted beam retraces the incident beam):

```julia
Λ = 1/(300e3)  # grating period (300 lines/mm)
m = -1          # diffraction order
θi = Fields.littrow_angle(λ0, Λ; m)
```

[`Fields.grating_GDD`](@ref) returns the GDD per unit separation (s²/m) using the
Treacy formula:

```julia
GDD_per_m = Fields.grating_GDD(λ0, Λ, m, θi)  # in s²/m
```

### Propagation

[`Fields.prop_gratings!`](@ref) / [`Fields.prop_gratings`](@ref) apply the exact
spectral phase of a four-grating compressor:

```julia
L = 5e-3  # 5 mm grating separation
Eω_out = Fields.prop_gratings(Eω, grid, Λ, L, m, θi, λ0)
```

Frequency components outside the grating diffraction bandwidth are automatically
set to zero.

### Optimised compression

[`Fields.optcomp_gratings`](@ref) finds the grating separation that maximises peak
power:

```julia
# With explicit search bounds
L_opt, Eω_comp = Fields.optcomp_gratings(Eω, grid, Λ, m, θi, 0.0, 0.05; λ0=λ0)

# Auto-estimated bounds (uses Taylor estimate internally)
L_opt, Eω_comp = Fields.optcomp_gratings(Eω, grid, Λ, m, θi; λ0=λ0)
```

### Example: SPM broadening with grating compression

The following example (from `examples/simple_interface/SPM_grating_compression.jl`)
simulates SPM broadening of a Yb-laser pulse in an Ar-filled HCF, followed by grating
compression:

```julia
using Luna

# Fibre parameters
a = 125e-6; flength = 1.0; gas = :Ar; pressure = 4.0

# Pulse parameters
λ0 = 1030e-9; τfwhm = 300e-15; energy = 600e-6

# Run simulation
output = prop_capillary(a, flength, gas, pressure;
    λ0, τfwhm, energy, λlims=(800e-9, 1300e-9), trange=2e-12)

# Grating compressor (300 lines/mm, order -1, Littrow)
Λ = 1/(300e3)
m = -1
θi = Fields.littrow_angle(λ0, Λ; m)

# Propagation function for compressed output
function prop(grid, Eω)
    L_opt, Eωcomp = Fields.optcomp_gratings(Eω, grid, Λ, m, θi; λ0=λ0)
    @info "Optimal grating separation: \$(round(L_opt*1000, digits=2)) mm"
    Eωcomp
end

# Plot compressed pulse
Plotting.time_1D(output; propagate=prop, trange=(-1000e-15, 1000e-15))
```

## Prism pair compression

A double-pass prism pair compressor provides tuneable negative GDD similarly to gratings,
but with lower losses (especially at Brewster incidence) and less higher-order dispersion
for moderate bandwidths. Prism compressors are widely used in Ti:sapphire and Yb-based
ultrafast laser systems.

The spectral phase is computed by **full 2D ray tracing** through both prisms.

### Geometry

The compressor consists of two identical prisms in an anti-parallel (apex-up / apex-down)
configuration. A retroreflecting mirror sends the beam back through both prisms, doubling
the spectral phase and eliminating spatial chirp.

The key parameters are:
- **`material`**: prism material (e.g. `:SiO2`, `:BK7`, `:CaF2`, `:MgF2`)
- **`α`**: apex angle of the prisms (radians)
- **`θi`**: angle of incidence on the first prism surface (radians)
- **Separation** (one of three options — see below)
- **`l1`, `l2`**: insertion of each prism (metres, along the prism face from the apex to the beam)

### Separation options

The prism separation can be specified in three equivalent ways. Provide **one**:

| Option | Description | Convention |
|--------|-------------|------------|
| `L` (positional) | Apex-to-apex Euclidean distance | Keller/Weiner |
| `L_lightcon` (keyword) | Perpendicular distance between Prism 1 input face and Prism 2 output face | Lightcon |
| `w`, `h` (keyword pair) | Horizontal / vertical apex displacements (`h` positive downward) | Direct |

Conversion: `L_lightcon = w cos(α/2) + h sin(α/2)`, `L_keller = sqrt(w² + h²)`.

When using `L` (Keller convention), `l2` is determined by the ray trace (matched to `l1`
at the center wavelength). When using `L_lightcon`, both `l1` and `l2` must be specified
explicitly.

### Insertion parameters

- **`l1`** (default `0.0`): distance **along the input face** of Prism 1, from the apex
  to the point where the beam hits. At `l1 = 0`, the beam grazes the apex and traverses
  no glass.
- **`l2`** (default `0.0`): distance along the entry face of Prism 2 from the apex.
  Only independently specified when using `L_lightcon`; otherwise determined by the
  ray trace.

### Dispersion mechanism

- **Angular dispersion** (from the separation): different wavelengths exit the first
  prism at different angles, travelling different optical path lengths to the second
  prism. This provides **negative** GDD. Increasing the separation increases the
  magnitude of the negative GDD.
- **Material traversal** (from insertions `l1`, `l2`): glass in the beam path adds
  **positive** GDD. Increasing `l1`/`l2` pushes the prism into the beam, adding more
  material and more positive GDD.

!!! note "Insertion and separation convention"
    With **`L_lightcon`**, `l1` and `l2` are independent: increasing either one adds
    glass to the beam path and shifts the net GDD towards more positive values. This
    matches the lab practice of translating a prism perpendicular to the beam.

    With **Keller `L`** (apex-to-apex distance), only `l1` is a free parameter — `l2`
    is determined by the ray trace. Changing `l1` shifts the beam entry point on
    Prism 1, but the exit point on Prism 2 adjusts to compensate, so the **total**
    glass path barely changes. To tune insertion with Keller `L`, adjust the
    apex-to-apex distance instead.

The net GDD can be tuned from strongly negative (large separation, small insertion) to
positive (small separation, large insertion).

### Single vs double pass

By default, all functions assume a **double pass** (retroreflected) geometry, which
doubles the spectral phase and eliminates spatial chirp. Set `double_pass=false` for a
single-pass geometry.

### Helper functions

[`Fields.brewster_angle`](@ref) calculates the Brewster angle for a given material and
wavelength:

```julia
θB = Fields.brewster_angle(:SiO2, 800e-9)   # ≈ 55.5° for fused silica
```

[`Fields.mindev_apex`](@ref) gives the apex angle for a Brewster-cut prism at minimum
deviation:

```julia
α = Fields.mindev_apex(:SiO2, 800e-9)   # ≈ 69.1° for fused silica
```

[`Fields.prism_pair_GDD`](@ref) computes the GDD per unit Keller separation (s²/m):

```julia
GDD_per_m = Fields.prism_pair_GDD(800e-9, :SiO2, α, θB)  # negative, in s²/m
```

### Propagation

[`Fields.prop_prisms!`](@ref) / [`Fields.prop_prisms`](@ref) apply the spectral phase
from full ray tracing through the prism pair:

```julia
α = Fields.mindev_apex(:SiO2, 800e-9)
θi = Fields.brewster_angle(:SiO2, 800e-9)
```

**Using Keller separation** (apex-to-apex distance):
```julia
L = 0.5   # 50 cm apex-to-apex
Eω_out = Fields.prop_prisms(Eω, grid, :SiO2, α, L, θi, 0.0, 0.0, λ0)
```

**Using Lightcon separation** (perpendicular face distance):
```julia
Eω_out = Fields.prop_prisms(Eω, grid, :SiO2, α, 0.0, θi, 10e-3, 10e-3, λ0;
                             L_lightcon=600e-3)
```

**Using direct apex displacements**:
```julia
Eω_out = Fields.prop_prisms(Eω, grid, :SiO2, α, 0.0, θi, 10e-3, 10e-3, λ0;
                             w=0.571, h=0.229)
```

**Single pass** (no retroreflection):
```julia
Eω_out = Fields.prop_prisms(Eω, grid, :SiO2, α, L, θi; double_pass=false)
```

Frequency components for which total internal reflection occurs inside the prisms are
automatically set to zero.

### Optimised compression

[`Fields.optcomp_prisms`](@ref) finds the prism separation that maximises peak power:

```julia
# With explicit search bounds (Keller separation)
L_opt, Eω_comp = Fields.optcomp_prisms(Eω, grid, :SiO2, α, θi, l1, l2,
                                        0.1, 2.0; λ0=800e-9)

# Auto-estimated bounds
L_opt, Eω_comp = Fields.optcomp_prisms(Eω, grid, :SiO2, α, θi; λ0=800e-9)
```

### Example: prism compression of a chirped pulse

```julia
using Luna

λ0 = 800e-9

# Brewster-cut fused silica prism pair
α = Fields.mindev_apex(:SiO2, λ0)
θi = Fields.brewster_angle(:SiO2, λ0)

# Create a positively chirped pulse (e.g. from SPM broadening)
grid = Grid.RealGrid(1, λ0, (500e-9, 1200e-9), 10e-12)
FT = FFTW.plan_rfft(Array{Float64}(undef, length(grid.t)), 1)
input = Fields.GaussField(λ0=λ0, τfwhm=50e-15, energy=1e-6)
Eω = input(grid, FT)
Eω_chirped = Fields.prop_taylor(Eω, grid, [0, 0, 500e-30], λ0)  # +500 fs² GDD

# Find optimal prism separation for compression
L_opt, Eω_compressed = Fields.optcomp_prisms(Eω_chirped, grid, :SiO2, α, θi; λ0=λ0)
@info "Optimal prism separation: $(round(L_opt*100, digits=1)) cm"
```

## Waveguide mode propagation

[`Fields.prop_mode!`](@ref) / [`Fields.prop_mode`](@ref Fields.prop_mode!) propagate the field along a
waveguide mode, including both dispersion and loss:

```julia
mode = Capillary.MarcatiliMode(a, :He, 1.0)
Eω_out = Fields.prop_mode(Eω, grid, mode, 1.0, λ0)  # 1 m propagation
```

## Combining elements

Multiple optical elements can be chained in a propagation function. The
`examples/simple_interface/2stage_compressor_RDWemission.jl` example shows how to
combine chirped mirrors, air path, glass windows, and silica wedge optimisation:

```julia
function prop!(Eω, grid)
    Fields.prop_mirror!(Eω, grid, 10, :PC70)          # 10 chirped mirror bounces
    Fields.prop_material!(Eω, grid, :Air, 5, λ0)      # 5 m of air path
    Fields.prop_material!(Eω, grid, :SiO2, 2e-3, λ0)  # 2 mm fused silica (windows)
    # Optimise compression with silica wedges
    _, Eωopt = Fields.optcomp_material(Eω, grid, :SiO2, λ0, -1e-2, 1e-2)
    Eω .= Eωopt
end
```

This `prop!` function can then be:
- used directly to post-process a field: `prop!(Eω, grid)`
- passed to plotting functions: `Plotting.time_1D(output; propagate=prop)`
- used with `PropagatedField` for multi-stage simulations:
  `pulse = Pulses.LunaPulse(output; energy, propagator=prop!)`

## API reference

For full details on all functions described above, see the auto-generated API
documentation for [Fields.jl](@ref).
