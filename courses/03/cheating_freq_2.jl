using PyPlot
using Gen

N = 100

X = 35

@gen function freq_cheating_model()

    freq_cheating ~ uniform(0, 1)
    
    p_skewed = 0.5 * freq_cheating + 0.25

    obs ~ binom(N, p_skewed)

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

trs = block_resimulation_inference(2500, 25000)

freq_cheating_samples = [trs[i][:freq_cheating] for i=1:25000]

hist(freq_cheating_samples, histtype="stepfilled", density=true, alpha=0.85, bins=30,
         label="posterior distribution", color="#348ABD")
vlines([.05, .35], [0, 0], [5, 5], alpha=0.3)
xlim(0, 1)
legend()
plt[:show]()
