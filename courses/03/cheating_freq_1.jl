using PyPlot
using Gen

N = 100

X = 35

@gen function freq_cheating_model()

    freq_cheating ~ uniform(0, 1)
    true_answers = [bernoulli(freq_cheating) for i = 1 : N]
    first_coin_flips = [bernoulli(0.5) for i = 1 : N]
    second_coin_flips = [bernoulli(0.5) for i = 1 : N]

    observed_proportion = sum(first_coin_flips .* true_answers .+ 
                              (1 .- first_coin_flips) .* second_coin_flips) / N

    obs ~ binom(N, observed_proportion)

end

function make_constraints()
    constraints = Gen.choicemap()
    constraints[:obs] = X
    constraints
end;

function block_resimulation_update(tr)

    # Block 1: Update freq_cheating
    latent_variable = select(:freq_cheating)
    (tr, _) = mh(tr, latent_variable)
    
    tr

end

function block_resimulation_inference(n_burnin, n_samples)
    observations = make_constraints()
    (tr, _) = generate(freq_cheating_model, (), observations)
    for iter=1:n_burnin
        tr = block_resimulation_update(tr)
    end
    trs = []
    for iter=1:n_samples
        tr = block_resimulation_update(tr)
        push!(trs, tr)
    end

    trs

end;

trs = block_resimulation_inference(15000, 35000)

freq_cheating_samples = [trs[i][:freq_cheating] for i=1:35000]

hist(freq_cheating_samples, histtype="stepfilled", density=true, alpha=0.85, bins=30,
         label="posterior distribution", color="#348ABD")
vlines([.05, .35], [0, 0], [5, 5], alpha=0.3)
xlim(0, 1)
legend()
plt[:show]()
