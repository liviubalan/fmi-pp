using PyPlot
using Gen

true_N = 500

D = rand(1:true_N, 10)

n_count_data = length(D)


@gen function german_tank_model()

    N ~ uniform_discrete(maximum(D), 10000)    

    obs = Int64[]

    for i = 1:n_count_data

        push!(obs, {(:obs, i)} ~ uniform_discrete(0,N))

    end

    obs

end

function make_constraints(ys::Vector{Int64})
    constraints = Gen.choicemap()
    for i=1:length(ys)
        constraints[(:obs, i)] = ys[i]
    end
    constraints
end;

function block_resimulation_update(tr)

    # Block 1: Update N
    latent_variable = select(:N)
    (tr, _) = mh(tr, latent_variable)
    
    tr

end

function block_resimulation_inference(n_burnin, n_samples)
    observations = make_constraints(D)
    (tr, _) = generate(german_tank_model, (), observations)
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

trs = block_resimulation_inference(10000, 30000)

N_samples = [trs[i][:N] for i=1:30000]

plt.hist(N_samples, density=true)
plt.show()
