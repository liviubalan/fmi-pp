using DelimitedFiles
using PyPlot
using Statistics
using Gen

challenger_data = readdlm("challenger_clean_data.csv", ',', Int64, '\n')

temperature = challenger_data[:, 1]
D = challenger_data[:, 2]  # defect or not?

n_count_data = length(D)

function logistic(x, beta, alpha)
    return 1.0 / (1.0 + exp(beta * x + alpha))
end

n_count_data = length(D)


@gen function logreg_model()


    beta ~ normal(0, 50)
    alpha ~ normal(0, 50)

    obs = Int64[]

    for i = 1:n_count_data

        push!(obs, {(:obs, i)} ~ bernoulli(logistic(temperature[i], beta, alpha)))

    end

    obs

end

function make_constraints(ys::Vector{Int64})
    constraints = Gen.choicemap()
    for i=1:length(ys)
        constraints[(:obs, i)] = D[i]
    end
    constraints
end;

function block_resimulation_update(tr)

    # Block 1: Update beta
    latent_variable = select(:beta)
    (tr, _) = mh(tr, latent_variable)

    # Block 2: Update alpha
    latent_variable = select(:alpha)
    (tr, _) = mh(tr, latent_variable)

    tr

end

function block_resimulation_inference(n_burnin, n_samples, thin)
    observations = make_constraints(D)
    (tr, _) = generate(logreg_model, (), observations)
    tr = map_optimize(tr, select(:beta, :alpha))
    for iter=1:n_burnin
        tr = block_resimulation_update(tr)
        println(iter)
    end
    trs = []
    for iter=1:n_samples
        for itert = 1:thin
            tr = block_resimulation_update(tr)
        end
        push!(trs, tr)
        println(iter)
    end

    trs

end;

trs = block_resimulation_inference(120000, 50000, 2)

beta_samples = [trs[i][:beta] for i=1:50000]
alpha_samples = [trs[i][:alpha] for i=1:50000]

# histogram of the samples:
plt.subplot(211)
plt.title("Posterior distributions of the variables \$\\alpha\$, \$\\beta\$")
plt.hist(beta_samples, histtype="stepfilled", bins=35, alpha=0.85,
         label="posterior of \$\\beta\$", color="#7A68A6", density=true)
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, histtype="stepfilled", bins=35, alpha=0.85,
         label="posterior of \$\\alpha\$", color="#A60628", density=true)
plt.legend()
plt.show()

prob_31 = [logistic(31, beta_samples[i], alpha_samples[i]) for i = 1:50000]

println(prob_31[1:10])

plt.xlim(0.9, 1.1)
plt.hist(prob_31, bins=50, histtype="stepfilled", density=true)
plt.title("Posterior distribution of probability of defect, given \$t = 31\$")
plt.xlabel("probability of defect occurring in O-ring")
plt.show()
