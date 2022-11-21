using PyPlot
using Gen

# The parameters to be inferred. We only know them here because we are synthesising the data.
true_alpha = 10
true_beta = 50

num_flashes = 5000

# Generate the angles
true_thetas = pi * rand(num_flashes) .- pi / 2

# Generate the x coordinates of the flashes along the coastline
data = true_alpha .+ true_beta * map(tan, true_thetas)

include("ObsDist.jl")

@gen function lighthouse_model()

    alpha ~ normal(0, 50)
    beta ~ exponential(1.0/100)

    obs = Float64[]

    for i = 1:num_flashes

        push!(obs, @trace(obsdist(alpha, beta), (:obs, i)))

    end

    obs

end

function make_constraints(ys::Vector{Float64})
    constraints = Gen.choicemap()
    for i=1:length(ys)
        constraints[(:obs, i)] = ys[i]
    end
    constraints
end;

function block_resimulation_update(tr)

    # Block 1: Update alpha
    latent_variable = select(:alpha)
    (tr, _) = mh(tr, latent_variable)

    # Block 2: Update beta
    latent_variable = select(:beta)
    (tr, _) = mh(tr, latent_variable)

    tr

end

function block_resimulation_inference(n_burnin, n_samples)
    observations = make_constraints(data)
    (tr, _) = generate(lighthouse_model, (), observations)
    for iter=1:n_burnin
        tr = block_resimulation_update(tr)
        println(iter)
    end
    trs = []
    for iter=1:n_samples
        tr = block_resimulation_update(tr)
        push!(trs, tr)
        println(iter)
    end

    trs

end;

trs = block_resimulation_inference(10000, 30000)

alpha_samples = [trs[i][:alpha] for i=1:30000]
beta_samples = [trs[i][:beta] for i=1:30000]

plt.hist(alpha_samples, density=true)
plt.show()

plt.hist(beta_samples, density=true)
plt.show()
