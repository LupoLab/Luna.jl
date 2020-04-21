# Modal decompositions
!!! note
    All Fourier transforms in this section are written in the optics sign convention. See [A note on sign conventions](@ref) for details on what is used in the code.

## Multi-mode guided
For propagation in waveguides taking into account multiple modes and the coupling between them, Luna uses the model laid out in [Kolesik and Moloney, *Nonlinear optical pulse propagation simulation: From Maxwell’s to unidirectional equations*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.70.036604) and [Tani et al., *Multimode ultrafast nonlinear optics in optical waveguides: numerical modeling and experiments in kagomé photonic-crystal fiber*](http://josab.osa.org/abstract.cfm?URI=josab-31-2-311). This is implemented in [`NonlinearRHS.TransModal`](@ref). The electric field ``\mathbf{E}(t, \mathbf{r_\perp}, z)`` is expressed as the inverse Fourier transform in time and the superposition of waveguide modes in space. This means that the transverse wave vector ``\mathbf{k}_\perp`` turns into a modal index ``j`` (this transform is implemented in [`Modes.ToSpace`](@ref)) and [`Modes.to_space!`](@ref)):
```math
\mathbf{E}(t, \mathbf{r_\perp}, z) = \frac{1}{2\pi} \int_{-\infty}^\infty \mathrm{d} \omega \sum_j \hat{\mathbf{e}}_j(\mathbf{r_\perp}, z) \tilde{E}_j(\omega, z) \mathrm{e}^{-i \omega t}\,,
```
where ``\hat{\mathbf{e}}_j(\mathbf{r_\perp}, z)`` is the orthonormal transverse field distribution of the ``j^{\mathrm{th}}`` mode and ``\tilde{E}_j(\omega, z)`` is the frequency-domain field in mode ``j``. The mode fields ``\hat{\mathbf{e}}_j(\mathbf{r_\perp}, z)`` are taken to be independent of frequency but can depend on the propagation coordinate ``z`` (e.g. in tapered waveguides). They can be vector quantities if polarisations other than purely lineary ``x``- or ``y``-polarisations need to be taken into account. The modes are normalised such that ``\vert \tilde{E}_j(\omega, z) \vert^2`` gives the spectral power density in mode ``j``, and equivalently ``\vert E_j(t, z)\vert^2`` gives the instantaneous power. The inverse transform is simply the overlap integral of the total field with each mode combined with the Fourier transform:
```math
\tilde{E}_j(\omega, z) = \int_S \mathrm{d}^2\mathbf{r_\perp} \int_{-\infty}^\infty \mathrm{d} t\,\, \hat{\mathbf{e}}_j^*(\mathbf{r_\perp}, z) \cdot \mathbf{E}(t, \mathbf{r_\perp}, z) \mathrm{e}^{i \omega t}\,,
```
where ``S`` is the cross-sectional area of the waveguide. This transform is implemented in [`NonlinearRHS.TransModal`](@ref) for use within simulations and in [`Modes.overlap`](@ref) for decomposition of existing sampled fields.

The linear operator for a mode ``\mathcal{L}_j(\omega, z)`` is given by (see [`LinearOps.make_const_linop`](@ref))
```math
\mathcal{L}_j(\omega, z) = i\left(\beta_j(\omega, z) - \frac{\omega}{v}\right) - \frac{1}{2}\alpha_j(\omega, z)\,,
```
where ``\beta_j(\omega, z)`` describes the phase evolution of the mode, ``v`` is a chosen frame velocity (this is the same for all modes) and ``\alpha(\omega, z)`` describes the attenuation of the waveguide (i.e. ``1/\alpha`` is the ``1/\mathrm{e}`` loss-length). This can also be expressed in terms of the *effective index* of the mode:
```math
\mathcal{L}_j(\omega, z) = i \left(\frac{\omega}{c} n_\mathrm{eff}(\omega, z) - \frac{\omega}{v}\right)\,,
```
where ``c`` is the speed of light in vacuum and ``n_\mathrm{eff}`` is complex, ``n_\mathrm{eff} = n + i k``, with ``n`` describing the effective refractive index and ``k`` describing the attenuation. With the modal power normalisation for ``\hat{\mathbf{e}}_j(\mathbf{r_\perp}, z)``, the normalisation factor ``N_{\mathrm{nl}}`` comes out as simply ``N_{\mathrm{nl}}=4``. The propagation equation, coupling the modes through the nonlinear polarisation, is therefore
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
- [`LinearOps.make_const_linop`](@ref)
- [`LinearOps.make_linop`](@ref)


## Single-mode guided

## Radially symmetric free-space

## Three-dimensional free-space