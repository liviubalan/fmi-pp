using Gen

struct ObsDist <: Gen.Distribution{AbstractVector{Real}} end

const obsdist = ObsDist()

function Gen.logpdf(::ObsDist, x::AbstractVector{Real}, alpha::Real, beta::Real, dim::Integer)

    return sum((log(beta) - log(pi)) .- map(log, beta ^ 2 .+ (alpha .- x) .^ 2))

end

function Gen.random(::ObsDist, alpha::Real, beta::Real, dim::Integer)

    return alpha .+ beta * map(tan, pi * rand(dim) .- pi / 2)

end

Gen.has_output_grad(::ObsDist) = false
Gen.has_argument_grads(::ObsDist) = (false, false, false, false)

function Gen.logpdf_grad(::ObsDist, x::AbstractVector{Real}, alpha::Real, beta::Real)
   
   (nothing, nothing, nothing, nothing)

end

(::ObsDist)(alpha, beta, dim) = random(ObsDist(), alpha, beta, dim)

