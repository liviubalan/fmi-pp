using Gen

struct ObsDist <: Gen.Distribution{Real} end

const obsdist = ObsDist()

function Gen.logpdf(::ObsDist, x::Real, alpha::Real, beta::Real)

    return log(beta) - log(pi) - log(beta ^ 2 + (alpha - x) ^ 2)

end

function Gen.random(::ObsDist, alpha::Real, beta::Real)

    return alpha + beta * tan(pi * rand() - pi / 2)

end

Gen.has_output_grad(::ObsDist) = false
Gen.has_argument_grads(::ObsDist) = (false, false, false)

function Gen.logpdf_grad(::ObsDist, x::Real, alpha::Real, beta::Real)
   
   (nothing, nothing, nothing, nothing)

end

(::ObsDist)(alpha, beta) = random(ObsDist(), alpha, beta)

