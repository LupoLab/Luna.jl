module Maths
import ForwardDiff

function derivative(f, x, order)
    if order == 0
        return f(x)
    elseif order == 1
        return ForwardDiff.derivative(f, x)
    else
        return derivative(x -> ForwardDiff.derivative(f, x), x, order-1)
    end
end

end