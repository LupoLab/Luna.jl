module Modes
import Luna:PhysData

function norm_mode_average(ω, βfun)
    out = zero(ω)
    function norm(z)
        out .= PhysData.c^2 * PhysData.ε_0 * βfun(ω, 1, 1, z) ./ ω
        return out
    end
    return norm
end

end