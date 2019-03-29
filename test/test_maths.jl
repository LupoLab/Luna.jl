import Test: @test
import Luna: Maths

f(x) = @. 4x^3 + 3x^2 + 2x + 1

@test Maths.derivative(f, 1, 1) == 12+6+2
@test Maths.derivative(f, 1, 2) == 24+6
@test Maths.derivative(f, 1, 3) == 24

e(x) = @. exp(x)

x = [1, 2, 3, 4, 5]
@test Maths.derivative(e, 1, 5) == exp(1)
@test Maths.derivative.(e, x, 5) == exp.(x)