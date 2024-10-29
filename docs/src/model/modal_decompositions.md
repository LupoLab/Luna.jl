# Modal decompositions
!!! note
    All Fourier transforms in this section are written in the optics sign convention. See [A note on sign conventions](@ref) for details on what is used in the code.

## Multi-mode guided
For propagation in waveguides taking into account multiple modes and the coupling between them, Luna uses the model laid out in [Kolesik and Moloney, *Nonlinear optical pulse propagation simulation: From Maxwell’s to unidirectional equations*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.70.036604) and [Tani et al., *Multimode ultrafast nonlinear optics in optical waveguides: numerical modeling and experiments in kagomé photonic-crystal fiber*](http://josab.osa.org/abstract.cfm?URI=josab-31-2-311). This is implemented in [`NonlinearRHS.TransModal`](@ref). The electric field ``\mathbf{E}(t, \mathbf{r_\perp}, z)`` is expressed as the inverse Fourier transform in time and the superposition of waveguide modes in space. This means that the transverse wave vector ``\mathbf{k}_\perp`` turns into a modal index ``j`` (this transform is implemented in [`Modes.ToSpace`](@ref) and [`Modes.to_space!`](@ref)):
```math
\mathbf{E}(t, \mathbf{r_\perp}, z) = \frac{1}{2\pi} \int_{-\infty}^\infty \mathrm{d} \omega \sum_j \hat{\mathbf{e}}_j(\mathbf{r_\perp}, z) \tilde{A}_j(\omega, z) \mathrm{e}^{-i \omega t}\,,
```
where ``\hat{\mathbf{e}}_j(\mathbf{r_\perp}, z)`` is the orthonormal transverse field distribution of the ``j^{\mathrm{th}}`` mode and ``\tilde{A}_j(\omega, z)`` is the frequency-domain amplitude in mode ``j``. The mode fields ``\hat{\mathbf{e}}_j(\mathbf{r_\perp}, z)`` are taken to be independent of frequency but can depend on the propagation coordinate ``z`` (e.g. in tapered waveguides). They can be vector quantities if polarisations other than purely lineary ``x``- or ``y``-polarisations need to be taken into account. The modes are normalised such that ``\vert \tilde{A}_j(\omega, z) \vert^2`` gives the spectral energy density in mode ``j`` (when also taking into account the normalisation of the FFT), and equivalently ``\vert A_j(t, z)\vert^2`` gives the instantaneous power. The forward transform to reciprocal space is simply the overlap integral of the total field with each mode combined with the Fourier transform in time:
```math
\tilde{A}_j(\omega, z) = \int_S \mathrm{d}^2\mathbf{r_\perp} \int_{-\infty}^\infty \mathrm{d} t\,\, \hat{\mathbf{e}}_j^*(\mathbf{r_\perp}, z) \cdot \mathbf{E}(t, \mathbf{r_\perp}, z) \mathrm{e}^{i \omega t}\,,
```
where ``S`` is the cross-sectional area of the waveguide. This transform is implemented in [`NonlinearRHS.TransModal`](@ref) for use within simulations and in [`Modes.overlap`](@ref) for decomposition of existing sampled fields. In both cases, the mode overlap integral is solved explicitly with a p-adaptive or h-adaptive cubature method.

The linear operator for a mode ``\mathcal{L}_j(\omega, z)`` is given by (see [`LinearOps.make_const_linop`](@ref))
```math
\mathcal{L}_j(\omega, z) = i\left(\beta_j(\omega, z) - \frac{\omega}{v}\right) - \frac{1}{2}\alpha_j(\omega, z)\,,
```
where ``\beta_j(\omega, z)`` is real-valued and describes the phase evolution of the mode, ``v`` is a chosen frame velocity (this is the same for all modes) and ``\alpha(\omega, z)`` (also real) describes the attenuation of the waveguide (i.e. ``1/\alpha`` is the ``1/\mathrm{e}`` power/energy loss length). This can also be expressed in terms of the *effective index* of the mode:
```math
\mathcal{L}_j(\omega, z) = i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\,,
```
where ``c`` is the speed of light in vacuum and ``n_\mathrm{eff}`` is complex, ``n_\mathrm{eff} = n + i k``, with ``n`` describing the effective refractive index and ``k`` describing the attenuation.

With the modal power normalisation for ``\hat{\mathbf{e}}_j(\mathbf{r_\perp}, z)``, the normalisation factor ``N_{\mathrm{nl}}`` comes out as simply ``N_{\mathrm{nl}}=4``. The propagation equation, coupling the modes through the nonlinear polarisation, is therefore
```math
\partial_z \tilde{A}_j(\omega, z) = i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\tilde{A}_j(\omega, z) + i\frac{\omega}{4} \tilde{\mathbf{P}}_\mathrm{nl}\,,
```
where ``\tilde{\mathbf{P}}_\mathrm{nl}`` is given by
```math
\tilde{\mathbf{P}}_\mathrm{nl} =  \int_S \mathrm{d}^2\mathbf{r_\perp} \int_{-\infty}^\infty \mathrm{d} t\,\, \hat{\mathbf{e}}_j^*(\mathbf{r_\perp}, z) \cdot \mathbf{P}_\mathrm{nl}\left[\mathbf{E}(t, \mathbf{r_\perp}, z)\right] \mathrm{e}^{i \omega t}
```
and ``\mathbf{E}(t, \mathbf{r_\perp}, z)`` is obtained from the set of ``\tilde{A}_j(\omega, z)`` as above.

The transverse coordinate ``\mathbf{r_\perp}`` for circular waveguides (e.g. hollow capillaries, optical fibres, and anti-resonant fibres) is in polar coordinates, ``\mathbf{r_\perp} = (r, \theta)``. For other waveguides (e.g. rectangular), it is Cartesian, ``\mathbf{r_\perp} = (x, y)``.

!!! note
    While ``\mathbf{r_\perp}`` can be given in either coordinate system, the **components** of the modal fields ``\hat{\mathbf{e}}_j(\mathbf{r_\perp}, z)`` are **always** given in Cartesian coordinates, i.e. the basis vectors for the polarisation of the field are always ``\mathbf{x}`` and ``\mathbf{y}``.

### Implementation
The modules and functions that define and implement this decomposition for different modes are
- [Modes.jl](@ref)
- [Capillary.jl](@ref)
- [RectModes.jl](@ref)
- [Antiresonant.jl](@ref)
- [`NonlinearRHS.TransModal`](@ref)
- [`NonlinearRHS.norm_modal`](@ref)
- [`LinearOps.make_const_linop`](@ref)
- [`LinearOps.make_linop`](@ref)


## Single-mode guided
In some situations, inter-mode coupling in a waveguide is negligible, so including several waveguide modes in the simulation unnecessarily slows down the computation. Simulating propagation in a single mode is trivially achieved by including only that single mode in both the forward and inverse transforms as defined above for [multi-mode propagation](#multi-mode-guided). For example, setting `modes=1` when calling `prop_capillary` achieves this and leads to a significant speed-up. However, in this simple implementation, the overlap integral between the nonlinear polarisation and the waveguide mode still needs to be calculated explicitly. We can make this unnecessary by making an assumption about the nonlinear polarisation.

If the nonlinear polarisation is *only due to third-order effects* like the Kerr effect or Raman scattering, we can express it as
```math
P_\mathrm{nl}\left(t, \mathbf{r}_\perp, z \right) = C\, E(t, \mathbf{r}_\perp, z)^3\,,
```
where ``C`` is a constant which depends on the specific effect (e.g. for the Kerr effect, ``C`` becomes ``\varepsilon_0 \chi^{(3)}`` with ``\chi^{(3)}`` the third-order susceptibility of the nonlinear medium) and we have switched to *explicitly real-valued* and *scalar* fields to make the notation simpler; the same result can be obtained with vector fields and more algebra. Expanding the field in terms of its modal content as above, this turns into
```math
P_\mathrm{nl}\left(t, \mathbf{r}_\perp, z \right) = C\, \Big[\sum_j \hat{e}_j(\mathbf{r_\perp}, z) A_j(t, z)\Big]^3\,,
```
where we have simply carried out the time-domain inverse Fourier transform to obtain ``A_j(t, z)``. For a single mode, this simplifies greatly to
```math
P_\mathrm{nl}\left(t, \mathbf{r}_\perp, z \right) = C\, \hat{e}_0(\mathbf{r_\perp}, z)^3 A(t, z)^3\,.
```
Now we can explicitly calculate the overlap integral with the single mode we are considering:
```math
\begin{align*}
P_\mathrm{nl}(t, z) &=  CA(t, z)^3\times\int_S \mathrm{d}^2\mathbf{r_\perp} \, \hat{e}_0^*(\mathbf{r_\perp}, z) \hat{e}_0(\mathbf{r_\perp}, z)^3\\[1em]
&= CA(t, z)^3\int_S \mathrm{d}^2\mathbf{r_\perp} \, \hat{e}_0(\mathbf{r_\perp}, z)^4\,,
\end{align*}
```
where ``A`` is now the modal amplitude in the single mode. In the second step we have made use of the fact that we are considering real-valued fields and hence ``\hat{e}_0(\mathbf{r_\perp}, z)`` is also real.

The mode normalisation in Luna is chosen such that the absolute value squared of the modal field amplitudes ``A_j(t, z)`` is the instantaneous power. For this to be fulfilled, we need
```math
\frac{1}{2} c \varepsilon_0 \int_S \mathrm{d}^2\mathbf{r_\perp} \left\vert \hat{e}_j(\mathbf{r_\perp}, z) \right\vert^2 = 1\,.
```
This, in turn, means that the *effective area* of the mode,
```math
A_{\mathrm{eff}, j}(z) = \frac{\left(\int_S \mathrm{d}^2\mathbf{r_\perp} \,\left\vert \hat{e}_j(\mathbf{r}_\perp, z)\right\vert^2\right)^2}{\int_S \mathrm{d}^2\mathbf{r_\perp}  \,\left\vert \hat{e}_j(\mathbf{r}_\perp, z)\right\vert^4}\,,
```
for the single mode we are considering is
```math
A_\mathrm{eff} = \Big(\frac{1}{4} c^2 \varepsilon_0^2 \int_S \mathrm{d}^2\mathbf{r_\perp} \, \hat{e}_0(\mathbf{r_\perp}, z)^4 \Big)^{-1}\,.
```
Note that ``A_\mathrm{eff}`` is **independent of the normalisation**, because the overall power of ``\hat{e}_j`` and any constants inside it is the same in the numerator and denominator. From this we can see that we can replace the integral expression in the projection of ``P_\mathrm{nl}`` with
```math
\int_S \mathrm{d}^2\mathbf{r_\perp} \, \hat{e}_0(\mathbf{r_\perp}, z)^4 = \frac{4}{\varepsilon_0^2c^2 A_\mathrm{eff}}\,.
```
We can now write down the **single-mode UPPE:**
```math

\partial_z \tilde{A}(\omega, z) = i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\tilde{A}(\omega, z)
 + i\frac{\omega}{4} C\frac{4}{\varepsilon_0^2c^2 A_\mathrm{eff}} \int_{-\infty}^\infty \mathrm{d} t\, A(t, z)^3 \mathrm{e}^{i \omega t}\,.
```
Crucially, the effective area depends on the mode shape ``\hat{e}_0(\mathbf{r_\perp}, z)``, but only needs to be calculated *once* (assuming the cross-section of the waveguide does not change along its length). When only third-order nonlinear effects are present, the single-mode UPPE is exactly equivalent to explicitly solving the projection integral, but *much* faster. However, it has two important drawbacks:

1. For obvious reasons, inter-modal coupling mediated by the nonlinearity is completely ignored.
2. The equation only works for third-order nonlinear effects, and hence photoionisation and plasma dynamics cannot be modelled in this way.

We can derive a different single-mode equation which can treat other nonlinear effects approximately. As written above, in the single-mode UPPE only the modal amplitude ``A`` appears. We now define a re-scaled **mode-averaged field** ``E_\mathrm{av}`` through
```math
A(t, z) = \sqrt{\frac{1}{2}\varepsilon_0 c A_\mathrm{eff}}E_\mathrm{av}(t, z)\,.
```
Note that, because ``\vert A(t, z)\vert^2`` has units of power, ``E_\mathrm{av}`` is in fact an electric field (with units of ``\mathrm{V/m}``). We can also define the *mode-averaged intensity* by
```math
I_\mathrm{av} = \frac{1}{2}\varepsilon_0 c \vert E_\mathrm{av}\vert^2 = \frac{\vert A(t, z)\vert^2}{A_\mathrm{eff}}\,.
```
Plugging in the definition of ``E_\mathrm{av}``, the UPPE reads
```math
\begin{align*}
\sqrt{\frac{1}{2}\varepsilon_0 c A_\mathrm{eff}}\partial_z \tilde{E}_\mathrm{av}(\omega, z) &= i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\sqrt{\frac{1}{2}\varepsilon_0 c A_\mathrm{eff}}\tilde{E}_\mathrm{av}(\omega, z)\\[1em]
&\qquad + i\frac{\omega}{4} C\Big(\frac{1}{2}\varepsilon_0 c A_\mathrm{eff}\Big)^{\frac{3}{2}}\frac{4}{\varepsilon_0^2c^2 A_\mathrm{eff}} \int_{-\infty}^\infty \mathrm{d} t\, E_\mathrm{av}(t, z)^3 \mathrm{e}^{i \omega t}\,.
\end{align*}
```
Cancelling the various constants, we arrive at the **mode-averaged field UPPE**
```math
\partial_z \tilde{E}_\mathrm{av}(\omega, z) = i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\tilde{E}_\mathrm{av}(\omega, z) + i\frac{\omega}{4} \frac{2}{\varepsilon_0 c} \int_{-\infty}^\infty \mathrm{d} t\, P_\mathrm{nl}\left[E_\mathrm{av}(t, z)\right] \mathrm{e}^{i \omega t}\,.
```
This now includes only a single inverse Fourier transform to obtain ``E_\mathrm{av}(t, z)`` followed by the calculation of ``P_\mathrm{nl}`` and then a forward transform. This equation is **still only valid for third-order responses** but we have now written it for an arbitrary polarisation ``P_\mathrm{nl}\left[E_\mathrm{av}(t, z)\right]``. Because ``E_\mathrm{av}`` is (a version of) the actual electric field, we can calculate arbitrary polarisation contributions, including the photoionisation and plasma term. However, this is still a significant approximation. For example, the mode-averaged intensity is approximately half of the on-axis intensity for the fundamental mode of a capillary fibre. Due to the exponential scaling of strong-field ionisation with intensity, this means that the peak ionisation fraction can be underestimated significantly. Only fully mode-resolved (and multi-mode) propagation can accurately model that situation.

The mode-averaged field UPPE as written above is very useful, but the scaling from ``A`` to ``E_\mathrm{av}`` changes the normalisation: ``\vert E_\mathrm{av}(t, z) \vert^2`` no longer gives the instantaneous power. To remain consistent with modal propagation simulations (e.g. for data analysis), Luna internally uses the same normalisation for both, which leads to a "hybrid" equation. The propagating quantity (and hence the simulation output) is ``A(z, t)`` and we switch to ``E_\mathrm{av}(z, t)`` to calculate the nonlinear polarisation. This leads to the appearance of an additional factor of in the equation
```math
\begin{align*}
\left(\frac{1}{2}\varepsilon_0 c A_\mathrm{eff}\right)^{-\frac{1}{2}}\partial_z \tilde{A}(\omega, z) &= i\left(\frac{1}{2}\varepsilon_0 c A_\mathrm{eff}\right)^{-\frac{1}{2}} \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\tilde{A}(\omega, z) + i\frac{\omega}{4} \frac{2}{\varepsilon_0 c} \int_{-\infty}^\infty \mathrm{d} t\, P_\mathrm{nl}\left[E_\mathrm{av}(t, z)\right] \mathrm{e}^{i \omega t}\\[1em]

\Rightarrow \partial_z\tilde{A}(\omega, z) &= i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\tilde{A}(\omega, z) + i\frac{\omega}{4}\sqrt{\frac{2A_\mathrm{eff}}{\varepsilon_0 c} }\int_{-\infty}^\infty \mathrm{d} t\, P_\mathrm{nl}\left[E_\mathrm{av}(t, z)\right] \mathrm{e}^{i \omega t}\,.
\end{align*}
```



## Radially symmetric free-space

## Three-dimensional free-space