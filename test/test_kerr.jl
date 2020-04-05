import Test: @test, @testset, @test_broken
import Luna: Maths, Nonlinear, PhysData
import FFTW

# test full / nothg / envelope kerr effect
Nt = collect(range(0, length=2^16))
t = @. (Nt - 2^16/2)*3.430944979182369e-16/4
E = exp.(-0.5.*(t./10e-15).^2).*cos.(2π*PhysData.c/800e-9.*t)
kerrfield = Nonlinear.Kerr_field(1.0)
outf = similar(E)
fill!(outf, 0.0)
kerrfield(outf, E)
kerrfieldn = Nonlinear.Kerr_field_nothg(1.0, 2^16)
outn = similar(E)
fill!(outn, 0.0)
kerrfieldn(outn, E)
kerrfieldenv = Nonlinear.Kerr_env(1.0)
Eenv = Maths.hilbert(E)
oute = similar(Eenv)
fill!(oute, 0.0)
kerrfieldenv(oute, Eenv)

outfω = FFTW.rfft(outf)
outnω = FFTW.rfft(outn)
outeω = FFTW.rfft(real.(oute))
# we compare only low (non THG) frequencies
# note that these are not expected to be exact, because we have dropped not just the THG term
# but also cross terms between positive and negative frequencies
@test isapprox(abs.(outnω[1800:2400]), abs.(outfω[1800:2400]), rtol=1e-15)
@test isapprox(abs.(outeω[1800:2400]), abs.(outfω[1800:2400]), rtol=1e-15)
