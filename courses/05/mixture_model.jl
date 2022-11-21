using DelimitedFiles
using PyPlot
using Statistics
using Distributions
using Gen

data = readdlm("mixture_data.csv", '\t', Float64, '\n')

n_data = size(data, 1)

data = vec(data)


@gen function clusters_model()

    p ~ uniform(0, 1)

    c1 ~ normal(120, 10)
    c2 ~ normal(190, 10)

    std1 ~ uniform(0, 100)
    std2 ~ uniform(0, 100)

    mixture_of_normals = HomogeneousMixture(normal, [0, 0])

    obs = Float64[]

    for i = 1:n_data

        push!(obs, @trace(mixture_of_normals([p, 1-p], [c1, c2], [std1, std2]), (:obs, i)))

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

    # Block 1: Update p
    latent_variable = select(:p)
    (tr, _) = mh(tr, latent_variable)

    # Block 2: Update c1
    latent_variable = select(:c1)
    (tr, _) = mh(tr, latent_variable)

    # Block 3: Update c2
    latent_variable = select(:c2)
    (tr, _) = mh(tr, latent_variable)

    # Block 4: Update std1
    latent_variable = select(:std1)
    (tr, _) = mh(tr, latent_variable)

    # Block 5: Update std2
    latent_variable = select(:std2)
    (tr, _) = mh(tr, latent_variable)

    tr

end

function block_resimulation_inference(tr, n_samples)
    trs = []
    for iter=1:n_samples
        tr = block_resimulation_update(tr)
        push!(trs, tr)
        println(iter)
    end

    trs

end;

#=

(tr, _) = generate(clusters_model, (), make_constraints(data))
tr = map_optimize(tr, select(:c1, :c2))
println(tr[:c1])
println(tr[:c2])

=#

observations = make_constraints(data)
(tr, _) = generate(clusters_model, (), observations)
tr = map_optimize(tr, select(:c1, :c2))
trs = block_resimulation_inference(tr, 50000)

c1_samples = [trs[i][:c1] for i=1:50000]
c2_samples = [trs[i][:c2] for i=1:50000]
std1_samples = [trs[i][:std1] for i=1:50000]
std2_samples = [trs[i][:std2] for i=1:50000]
p_samples = [trs[i][:p] for i=1:50000]


plt.subplot(311)
plt.plot(c1_samples, label="trace of center 1", c="#348ABD", lw=1)
plt.plot(c2_samples, label="trace of center 2", c="#A60628", lw=1)
plt.title("Traces of unknown parameters")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.7)

plt.subplot(312)
plt.plot(std1_samples, label="trace of of standard deviation of cluster 1", c="#348ABD", lw=1)
plt.plot(std2_samples, label="trace of of standard deviation of cluster 2", c="#A60628", lw=1)
plt.legend(loc="upper left")

plt.subplot(313)
plt.plot(p_samples, label="p: frequency of assignment to cluster 1", color="#467821", lw=1)
plt.xlabel("Steps")
#plt.ylim(0, 1)
plt.legend();

plt.show()

new_trs = block_resimulation_inference(trs[end], 100000)

c1_new_samples = [new_trs[i][:c1] for i=1:100000]
c2_new_samples = [new_trs[i][:c2] for i=1:100000]
std1_new_samples = [new_trs[i][:std1] for i=1:100000]
std2_new_samples = [new_trs[i][:std2] for i=1:100000]

plt.plot(collect(1:50000), c1_samples, label="previous trace of center 1", lw=1, alpha=0.4, c="#348ABD")
plt.plot(collect(1:50000), c2_samples, label="previous trace of center 2", lw=1, alpha=0.4, c="#A60628")

plt.plot(collect(50001:150000), c1_new_samples, label="new trace of center 1", lw=1, c="#348ABD")
plt.plot(collect(50001:150000), c2_new_samples, label="new trace of center 2", lw=1, c="#A60628")

plt.title("Traces of unknown center parameters")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.8)
plt.xlabel("Steps");

plt.show()

plt.subplot(2, 2, 1)
plt.title("Posterior of center of cluster 1")
plt.hist(c1_new_samples, color="#348ABD", bins=30, histtype="stepfilled")

plt.subplot(2, 2, 3)
plt.title("Posterior of center of cluster 2")
plt.hist(c2_new_samples, color="#A60628", bins=30, histtype="stepfilled")

plt.subplot(2, 2, 2)
plt.title("Posterior of standard deviation of cluster 1")
plt.hist(std1_new_samples, color="#348ABD", bins=30, histtype="stepfilled")

plt.subplot(2, 2, 4)
plt.title("Posterior of standard deviation of cluster 2")
plt.hist(std2_new_samples, color="#A60628", bins=30, histtype="stepfilled")

# plt.autoscale(tight=True)
plt.tight_layout()

plt.show()

x = 175
p_new_samples = [new_trs[i][:p] for i=1:100000]

v = p_new_samples .* [Distributions.pdf(Distributions.Normal(c1_new_samples[i], std1_new_samples[i]), x) for i=1:100000] >
    (1 .- p_new_samples) .* [Distributions.pdf(Distributions.Normal(c2_new_samples[i], std2_new_samples[i]), x) for i=1:100000]

println("Probability of belonging to cluster 1:", mean(v))   

