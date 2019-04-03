module Maths
import ForwardDiff

function derivative(f, x, order::Integer)
    if order == 0
        return f(x)
    elseif order == 1
        return ForwardDiff.derivative(f, x)
    else
        return derivative(x -> ForwardDiff.derivative(f, x), x, order-1)
    end
end

function gauss(x, σ; x0=0, power::Integer=2)
    return @. exp(-1//2*((x-x0)/σ)^power)
end

function gauss(x; x0=0, power::Integer=2, fwhm)
    σ = fwhm/(2*(2*log(2))^(1/power))
    return gauss(x, σ, x0=x0, power=power)
end

function moment(x::Array{T, 1}, y::Array{T, 1}, n=1) where T
    if length(x) ≠ length(y)
        throw(DomainError(x, "x and y must have same length"))
    end
    return sum(x.^n .* y)/sum(y)
end

function moment(x::Array{T, 1}, y, n=1; dim=1) where T
    if size(y, dim) ≠ length(x)
        throw(DomainError(y, "y must be of same length as x along dim"))
    end
    xshape = ones(Integer, ndims(y))
    xshape[dim] = length(x)
    return sum(reshape(x, Tuple(xshape)).^n .* y, dims=dim)./sum(y, dims=dim)
end

function rms_width(x::Array{T, 1}, y::Array{T, 1}; dim=1) where T
    return sqrt(moment(x, y, 2) - moment(x, y, 1)^2)
end

function rms_width(x::Array{T, 1}, y; dim=1) where T
    return sqrt.(moment(x, y, 2, dim=dim) - moment(x, y, 1, dim=dim).^2)
end

function cumtrapz!(ret, x, y)
    fill!(ret, 0)
    for i in 2 : length(y)
        ret[i] = ret[i-1] + (y[i-1] + y[i])*(x[i] - x[i-1])
    end
    ret *= 1//2
end

function cumtrapz(x, y)
    ret = similar(y)
    fill!(ret, 0)
    for i in 2 : length(y)
        ret[i] = ret[i-1] + (y[i-1] + y[i])*(x[i] - x[i-1])
    end
    ret *= 1//2
end

function normmax(x; dims=1)
    return x./maximum(x; dims=dims)
end

end