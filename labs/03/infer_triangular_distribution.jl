using PyPlot
using Gen

N = 100

include("Triangular.jl")

@gen function triangular_model(l, h, m)

    z = triangular(l, h, m)

end

traces = [Gen.simulate(triangular_model, (-3, 8, 0)) for _=1:N];

data = [traces[i][] for i=1:N]

@gen function model()

    l ~ uniform(-6, 0)
    h ~ uniform(5, 10)
    m ~ uniform(1, 4)

    obs_data = Float64[]

    for i = 1:N

        push!(obs_data, @trace(triangular_model(l, h, m), (:z, i)))

    end

    obs_data

end


function make_constraints(ys::Vector{Float64})
    constraints = Gen.choicemap()
    for i=1:length(ys)
        constraints[(:obs_data, i)] = ys[i]
    end
    constraints
end;

function block_resimulation_update(tr)

    # Block 1: Update l
    latent_variable = select(:l)
    (tr, _) = mh(tr, latent_variable)

    # Block 2: Update h
    latent_variable = select(:h)
    (tr, _) = mh(tr, latent_variable)

    # Block 3: Update m
    latent_variable = select(:m)
    (tr, _) = mh(tr, latent_variable)
    
    tr

end

function block_resimulation_inference(n_burnin, n_samples)
    observations = make_constraints(data)
    (tr, _) = generate(model, (), observations)
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

l_samples = [trs[i][:l] for i=1:30000]
h_samples = [trs[i][:h] for i=1:30000]
m_samples = [trs[i][:m] for i=1:30000]



# histogram of the samples:

hist(l_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label="posterior of l", color="#A60628")
plt[:show]()

hist(h_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label="posterior of h", color="#A60628")
plt[:show]()

hist(m_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label="posterior of m", color="#A60628")
plt[:show]()

