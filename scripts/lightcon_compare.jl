using Luna; using Luna: Fields, PhysData, Maths

λ0 = 1030e-9
ω0 = 2π * PhysData.c / λ0

# === 1. Material check ===
n_func = PhysData.ref_index_fun(:SiO2)
n0 = real(n_func(λ0))
println("=== Material properties at λ=1030nm ===")
println("n(SiO2) = $n0")
println("Brewster angle = $(rad2deg(atan(n0)))°  (Lightcon: 55.409°)")
println("Min-dev apex   = $(rad2deg(Fields.mindev_apex(:SiO2, λ0)))°  (Lightcon: 69.183°)")

# GVD = d²k/dω²  (units: s²/m)
# Convert: s²/m → fs²/mm:  ×1e30 (s²→fs²) × 1e-3 (1/m → 1/mm) = ×1e27
function k_at(ω_val)
    λv = PhysData.wlfreq(ω_val)
    nv = real(n_func(λv))
    ω_val * nv / PhysData.c
end
β2 = Maths.derivative(k_at, ω0, 2)
β3 = Maths.derivative(k_at, ω0, 3)
println("Material GVD = $(β2 * 1e27) fs²/mm  (Lightcon: 18.961)")
println("Material TOD = $(β3 * 1e42) fs³/mm  (Lightcon: 41.165)")

# === 2. Prism pair GDD per unit L (our convention: apex-to-apex) ===
α_lc = deg2rad(69.183)
θi_lc = deg2rad(55.409)
GDD_per_m = Fields.prism_pair_GDD(λ0, :SiO2, α_lc, θi_lc)
println("\n=== Prism pair GDD ===")
println("Our GDD/m (double-pass, apex-to-apex L) = $(GDD_per_m * 1e30) fs²/m")

# === 3. Geometry conversion ===
# From the SVG/Lightcon definition:
# L_ff = perpendicular distance between face 1 of P1 and face 2 of P2 (parallel faces)
# L_apex = apex-to-apex distance (our convention)
#
# The exit beam direction from P1 (= apex-to-apex direction) is:
#   exit_dir = (-cos(θe+α), sin(θe+α))
# in coordinates where face 1 outward normal = (-1,0).
# The x-component = -cos(θe+α). At Brewster/mindev: θe+α = π-θi, so x = cos(θi).
# Therefore: L_ff = L_apex × cos(θi)

θe1, _ = Fields._prism_trace(θi_lc, α_lc, n0)
exit_x = -cos(θe1 + α_lc)  # should equal cos(θi) at min dev
println("\n=== Geometry conversion ===")
println("exit_x = $exit_x,  cos(θi) = $(cos(θi_lc))  (should match at min dev)")
println("L_ff = L_apex × cos(θi)")

L_ff = 0.600  # 600 mm Lightcon
L_apex = L_ff / cos(θi_lc)
println("L_ff = $(L_ff*1e3) mm → L_apex = $(L_apex*1e3) mm")

# l conversion: Lightcon l = along face; our l = perpendicular to bisector
# l_ours = l_face × sin(α/2)
l_face = 10e-3  # 10 mm Lightcon
l_ours = l_face * sin(α_lc/2)
println("l_face = $(l_face*1e3) mm → l_ours = $(l_ours*1e3) mm")

# === 4. Compare GDD (air-only, no glass) at various L ===
println("\n=== GDD comparison (double-pass, air-only) ===")
GDD_dp_apex = GDD_per_m * L_apex
println("Our GDD (double-pass, L_apex=$(round(L_apex*1e3, digits=1))mm) = $(GDD_dp_apex * 1e30) fs²")
println("Our GDD (single-pass, L_apex) = $(GDD_dp_apex/2 * 1e30) fs²")
GDD_dp_ff = GDD_per_m * L_ff
println("Our GDD (double-pass, L=L_ff=600mm) = $(GDD_dp_ff * 1e30) fs²")
println("Our GDD (single-pass, L=L_ff=600mm) = $(GDD_dp_ff/2 * 1e30) fs²")
println("Lightcon GDD = -543.513 fs²")

# === 5. Full phase calculation with glass insertion ===
println("\n=== Full phase including glass insertion ===")
# With L_apex
phase_func_apex(ω_val) = Fields._prism_pair_phase(
    [ω_val], n_func, α_lc, L_apex, θi_lc, l_ours, l_ours)[1]
phase_func_apex_l0(ω_val) = Fields._prism_pair_phase(
    [ω_val], n_func, α_lc, L_apex, θi_lc, 0.0, 0.0)[1]

GDD_full_apex = Maths.derivative(phase_func_apex, ω0, 2)
TOD_full_apex = Maths.derivative(phase_func_apex, ω0, 3)
GDD_air_apex = Maths.derivative(phase_func_apex_l0, ω0, 2)

println("With L_apex=$(round(L_apex*1e3, digits=1))mm, l=$(round(l_ours*1e3, digits=2))mm:")
println("  GDD (total) = $(GDD_full_apex * 1e30) fs²  (Lightcon: -543.513)")
println("  TOD (total) = $(TOD_full_apex * 1e45) fs³  (Lightcon: 2204.753)")
println("  GDD (air)   = $(GDD_air_apex * 1e30) fs²")
println("  GDD (glass) = $((GDD_full_apex - GDD_air_apex) * 1e30) fs²")

# With L_ff (treating L_ff directly as apex-to-apex)
phase_func_ff(ω_val) = Fields._prism_pair_phase(
    [ω_val], n_func, α_lc, L_ff, θi_lc, l_ours, l_ours)[1]

GDD_full_ff = Maths.derivative(phase_func_ff, ω0, 2)
TOD_full_ff = Maths.derivative(phase_func_ff, ω0, 3)

println("\nWith L=L_ff=600mm (as if apex-to-apex), l=$(round(l_ours*1e3, digits=2))mm:")
println("  GDD (total) = $(GDD_full_ff * 1e30) fs²  (Lightcon: -543.513)")
println("  TOD (total) = $(TOD_full_ff * 1e45) fs³  (Lightcon: 2204.753)")

# === 6. Keller analytical formula ===
dndλ = Maths.derivative(λv -> real(n_func(λv)), λ0, 1)
println("\n=== Keller analytical formula ===")
println("dn/dλ = $(dndλ * 1e-6) per μm = $(dndλ) per m")
# Keller Eq. 3.20: for double-pass at Brewster/mindev:
# GDD = -4λ³L(dn/dλ)² / (πc²)
GDD_keller_dp_per_m = -4 * λ0^3 * dndλ^2 / (π * PhysData.c^2)
GDD_keller_sp_per_m = GDD_keller_dp_per_m / 2
println("Keller GDD/m (double-pass) = $(GDD_keller_dp_per_m * 1e30) fs²/m")
println("Keller GDD/m (single-pass) = $(GDD_keller_sp_per_m * 1e30) fs²/m")
println("Our    GDD/m (double-pass) = $(GDD_per_m * 1e30) fs²/m")
println("Ratio our/Keller = $(GDD_per_m / GDD_keller_dp_per_m)")

# What Keller gives at various L
println("\nKeller GDD values:")
println("  double-pass, L_apex: $(GDD_keller_dp_per_m * L_apex * 1e30) fs²")
println("  single-pass, L_apex: $(GDD_keller_sp_per_m * L_apex * 1e30) fs²")
println("  double-pass, L_ff:   $(GDD_keller_dp_per_m * L_ff * 1e30) fs²")
println("  single-pass, L_ff:   $(GDD_keller_sp_per_m * L_ff * 1e30) fs²")
println("  Lightcon says:        -543.513 fs²")

# === 7. What L gives the Lightcon answer? ===
println("\n=== Reverse-engineering Lightcon L ===")
L_needed_dp = -543.513e-30 / GDD_per_m
L_needed_sp = -543.513e-30 / (GDD_per_m/2)
println("L needed for our double-pass: $(L_needed_dp * 1e3) mm")
println("L needed for our single-pass: $(L_needed_sp * 1e3) mm")
println("L_ff = 600 mm")
println("Ratio L_ff / L_needed_dp = $(0.6 / L_needed_dp)")
println("Ratio L_ff / L_needed_sp = $(0.6 / L_needed_sp)")
