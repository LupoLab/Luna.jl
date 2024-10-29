# Modal decompositions
!!! note
    All Fourier transforms in this section are written in the optics sign convention. See [A note on sign conventions](@ref) for details on what is used in the code.

## Multi-mode guided
For propagation in waveguides taking into account multiple modes and the coupling between them, Luna uses the model laid out in [Kolesik and Moloney, *Nonlinear optical pulse propagation simulation: From Maxwell’s to unidirectional equations*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.70.036604) and [Tani et al., *Multimode ultrafast nonlinear optics in optical waveguides: numerical modeling and experiments in kagomé photonic-crystal fiber*](http://josab.osa.org/abstract.cfm?URI=josab-31-2-311). This is implemented in [`NonlinearRHS.TransModal`](@ref). The electric field ``\mathbf{E}(t, \mathbf{r_\perp}, z)`` is expressed as the inverse Fourier transform in time and the superposition of waveguide modes in space. This means that the transverse wave vector ``\mathbf{k}_\perp`` turns into a modal index ``j`` (this transform is implemented in [`Modes.ToSpace`](@ref) and [`Modes.to_space!`](@ref)):
```math
\mathbf{E}(t, \mathbf{r_\perp}, z) = \frac{1}{2\pi} \int_{-\infty}^\infty \mathrm{d} \omega \sum_j \hat{\mathbf{e}}_j(\mathbf{r_\perp}, z) \tilde{E}_j(\omega, z) \mathrm{e}^{-i \omega t}\,,
```
where ``\hat{\mathbf{e}}_j(\mathbf{r_\perp}, z)`` is the orthonormal transverse field distribution of the ``j^{\mathrm{th}}`` mode and ``\tilde{E}_j(\omega, z)`` is the frequency-domain amplitude in mode ``j``. The mode fields ``\hat{\mathbf{e}}_j(\mathbf{r_\perp}, z)`` are taken to be independent of frequency but can depend on the propagation coordinate ``z`` (e.g. in tapered waveguides). They can be vector quantities if polarisations other than purely lineary ``x``- or ``y``-polarisations need to be taken into account. The modes are normalised such that ``\vert \tilde{E}_j(\omega, z) \vert^2`` gives the spectral energy density in mode ``j`` (when also taking into account the normalisation of the FFT), and equivalently ``\vert E_j(t, z)\vert^2`` gives the instantaneous power. The forward transform to reciprocal space is simply the overlap integral of the total field with each mode combined with the Fourier transform in time:
```math
\tilde{E}_j(\omega, z) = \int_S \mathrm{d}^2\mathbf{r_\perp} \int_{-\infty}^\infty \mathrm{d} t\,\, \hat{\mathbf{e}}_j^*(\mathbf{r_\perp}, z) \cdot \mathbf{E}(t, \mathbf{r_\perp}, z) \mathrm{e}^{i \omega t}\,,
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
\partial_z \tilde{E}_j(\omega, z) = i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\tilde{E}_j(\omega, z) + i\frac{\omega}{4} \tilde{\mathbf{P}}_\mathrm{nl}\,,
```
where ``\tilde{\mathbf{P}}_\mathrm{nl}`` is given by
```math
\tilde{\mathbf{P}}_\mathrm{nl} =  \int_S \mathrm{d}^2\mathbf{r_\perp} \int_{-\infty}^\infty \mathrm{d} t\,\, \hat{\mathbf{e}}_j^*(\mathbf{r_\perp}, z) \cdot \mathbf{P}_\mathrm{nl}\left[\mathbf{E}(t, \mathbf{r_\perp}, z)\right] \mathrm{e}^{i \omega t}
```
and ``\mathbf{E}(t, \mathbf{r_\perp}, z)`` is obtained from the set of ``\tilde{E}_j(\omega, z)`` as above.

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
P_\mathrm{nl}\left(t, \mathbf{r}_\perp, z \right) = C\, \Big[\sum_j \hat{e}_j(\mathbf{r_\perp}, z) E_j(t, z)\Big]^3\,,
```
where we have simply carried out the time-domain inverse Fourier transform to obtain ``E_j(t, z)``. For a single mode (``j=0`` only), this simplifies greatly to
```math
P_\mathrm{nl}\left(t, \mathbf{r}_\perp, z \right) = C\, \hat{e}_0(\mathbf{r_\perp}, z)^3 E_0(t, z)^3\,.
```
Now we can explicitly calculate the overlap integral with the single mode we are considering:
```math
\begin{align*}
P_\mathrm{nl}(t, z) &=  CE_0(t, z)^3\times\int_S \mathrm{d}^2\mathbf{r_\perp} \, \hat{e}_0^*(\mathbf{r_\perp}, z) \hat{e}_0(\mathbf{r_\perp}, z)^3\\
&= CE_0(t, z)^3\int_S \mathrm{d}^2\mathbf{r_\perp} \, \hat{e}_0(\mathbf{r_\perp}, z)^4\\
&\equiv CE_0(t, z)^3 \Gamma\,,
\end{align*}
```
where in the second step we have made use of the fact that we are considering real-valued fields and hence ``\hat{e}_0(\mathbf{r_\perp}, z)`` is also real. The constant ``\Gamma`` depends on the mode shape ``\hat{e}_0(\mathbf{r_\perp}, z)``, but crucially, only needs to be calculated *once*. If we now define a re-scaled **mode-averaged** modal field ``E'`` through
```math
E_0(t, z) = \sqrt{\frac{2}{\varepsilon_0 c \Gamma}}E'(t, z)\,,
```
then the UPPE reads
```math
\sqrt{\frac{2}{\varepsilon_0 c \Gamma}}\partial_z \tilde{E}'(\omega, z) = i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\sqrt{\frac{2}{\varepsilon_0 c \Gamma}}\tilde{E}'(\omega, z) + i\frac{\omega}{4} C\Big(\frac{2}{\varepsilon_0 c\Gamma}\Big)^{\frac{3}{2}}\Gamma \int_{-\infty}^\infty \mathrm{d} t\, E'(t, z)^3 \mathrm{e}^{i \omega t}\,.
```
The factors of ``\Gamma`` and ``\Gamma^{-\frac{3}{2}}`` in the final term combine to cancel with the other ``\Gamma^{-\frac{1}{2}}`` terms, so that we arrive at the **mode-averaged UPPE**
```math
\partial_z \tilde{E}'(\omega, z) = i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\tilde{E}'(\omega, z) + i\frac{\omega}{4} \frac{2}{\varepsilon_0 c} \int_{-\infty}^\infty \mathrm{d} t\, P_\mathrm{nl}\left[E'(t, z)\right] \mathrm{e}^{i \omega t}\,.
```
This now includes only a single inverse Fourier transform to obtain ``E'(t, z)`` followed by the calculation of ``P_\mathrm{nl}`` and then a forward transform. However, note that we have played a trick in this last step of the derivation: this equation is **only valid for third-order responses** but we have now written it for an arbitrary polarisation ``P_\mathrm{nl}\left[E'(t, z)\right]``. The above derivation quickly fails for other polarisation types, most importantly photoionisation. That means that mode-averaged propagation is a *significant* approximation whenever photoionisation and plasma effects are important.

The mode-averaged UPPE as written above is very useful, but the scaling from ``E`` to ``E'`` changes the normalisation: ``\vert E'(t, z) \vert^2`` no longer gives the instantaneous power. To remain consistent with modal propagation simulations (e.g. for data analysis), Luna internally uses the same normalisation for both, that is, the propagating field is ``E(z, t)`` and we only switch to ``E'(z, t)`` to calculate the nonlinear polarisation. This leads to the appearance of an additional factor of ``\sqrt{\Gamma}`` in the equation
```math
\begin{align*}
\partial_z \sqrt{\Gamma}\tilde{E}(\omega, z) &= i\sqrt{\Gamma} \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\tilde{E}(\omega, z) + i\frac{\omega}{4} \int_{-\infty}^\infty \mathrm{d} t\, P_\mathrm{nl}\left[E'(t, z)\right] \mathrm{e}^{i \omega t}\\
\Rightarrow \partial_z\tilde{E}(\omega, z) &= i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\tilde{E}(\omega, z) + i\frac{\omega}{4\sqrt{\Gamma}} \int_{-\infty}^\infty \mathrm{d} t\, P_\mathrm{nl}\left[E'(t, z)\right] \mathrm{e}^{i \omega t}\,.
\end{align*}
```

### Connection to the effective area
The mode normalisation in Luna is chosen such that the absolute value squared of the modal field amplitudes ``E_j(t, z)`` is the instantaneous power. For this to be fulfilled, we need
```math
\frac{1}{2} c \varepsilon_0 \int_S \mathrm{d}^2\mathbf{r_\perp} \left\vert \hat{e}_j(\mathbf{r_\perp}, z) \right\vert^2 = 1\,.
```
This, in turn, means that the *effective area* of the mode,
```math
A_{\mathrm{eff}, j}(z) = \frac{\left(\int_S \mathrm{d}^2\mathbf{r_\perp} \,\left\vert \hat{e}_j(\mathbf{r}_\perp, z)\right\vert^2\right)^2}{\int_S \mathrm{d}^2\mathbf{r_\perp}  \,\left\vert \hat{e}_j(\mathbf{r}_\perp, z)\right\vert^4}\,,
```
is given by
```math
A_\mathrm{eff} = \Big(\frac{1}{4} c^2 \varepsilon_0^2 \Gamma \Big)^{-1}
```
with the scaling constant ``\Gamma`` as defined above. Note that ``A_\mathrm{eff}`` is **independent of the normalisation**, because the overall power of ``\hat{e}_j`` and any constants inside it is the same in the numerator and denominator. Hence the scaling factor can be obtained from the effective area as
```math
\Gamma = \Big(\frac{1}{4} c^2 \varepsilon_0^2 A_\mathrm{eff} \Big)^{-1}\,.
```

## Radially symmetric free-space

## Three-dimensional free-space