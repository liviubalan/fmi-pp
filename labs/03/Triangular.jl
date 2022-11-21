using Distributions
using Gen


struct Triangular <: Gen.Distribution{Real} end

const triangular = Triangular()

function logpdf(::Triangular, x::Real, low::Real, high::Real, mod::Real)

        if low <= x < mod
            return log((2. * (x - low)) / ((high - low) * (mod - low)))
        elseif x == c
            return log(2. / (high - low))
        elseif c < x <= b
            return log((2. * (high - x)) / ((high - low) * (high - mod)))
        end
        return log(0)

end

function random(::Triangular, low::Real, high::Real, mod::Real)
    rand(Distributions.TriangularDist(low, high, mod))
end

has_output_grad(::Triangular) = false
has_argument_grads(::Triangular) = (false, false, false)

function logpdf_grad(::Triangular, x::Real, low::Real, high::Real, mod::Real)
   
   (nothing, nothing, nothing, nothong)

end

(::Triangular)(low, high, mod) = random(Triangular(), low, high, mod)

