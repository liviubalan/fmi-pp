using DelimitedFiles
using PyPlot
using Statistics
using Gen

count_data = readdlm("txtdata.csv", '\t', Float64, '\n')
#count_data = readdlm("txtdata_sim.csv", '\t', Float64, '\n')

n_count_data = size(count_data, 1)

count_data = vec(count_data)

#=

#fig1 = figure("data", figsize = (12.5, 3.5))
bar(1:n_count_data, count_data, color="#348ABD")
xlabel("Time (days)")
ylabel("count of text-msgs received")
title("Did the user's texting habits change over time?")
xlim(0, n_count_data)
plt[:show]()

print(mean(count_data))

=#

alpha = 1.0 / mean(count_data)  # Recall count_data is the
                                # variable that holds our txt counts

@gen function switch_point_model(n_count_data::Integer, alpha::Real)

    lambda_1 ~ exponential(alpha) # syntactic sugar for:
                                  # lambda_1 = {:lambda_1} ~ exponential(alpha)
    lambda_2 ~ exponential(alpha)
    tau ~ uniform_discrete(0, n_count_data)

    cnt_data = Float64[]

    for i = 1:n_count_data

        if i < tau
           push!(cnt_data, {(:cnt_data, i)} ~ poisson(lambda_1))
        else
           push!(cnt_data, {(:cnt_data, i)} ~ poisson(lambda_2))
        end

    end

    cnt_data

end

function make_constraints(ys::Vector{Float64})
    constraints = Gen.choicemap()
    for i=1:length(ys)
        constraints[(:cnt_data, i)] = ys[i]
    end
    constraints
end;

#=

function block_resimulation_update(tr)

    # Block 1: Update the nodels's latent variables
    latent_variables = select(:tau, :lambda_1, :lambda_2)
    (tr, _) = mh(tr, latent_variables)
    
    tr

end

=#

function block_resimulation_update(tr)

    # Block 1: Update tau
    latent_variable = select(:tau)
    (tr, _) = mh(tr, latent_variable)

    # Block 2: Update lambda_1
    latent_variable = select(:lambda_1)
    (tr, _) = mh(tr, latent_variable)

    # Block 3: Update lambda_2
    latent_variable = select(:lambda_2)
    (tr, _) = mh(tr, latent_variable)
    
    tr

end

function block_resimulation_inference(n_count_data, alpha, count_data, n_burnin, n_samples)
    # fix observed data
    observations = make_constraints(count_data)
    # generate feasible starting point
    (tr, _) = generate(switch_point_model, (n_count_data, alpha), observations)

    # throw to garbage (can be entirely irrelevant if the starting point is far away from the posterior)
    for iter=1:n_burnin
        tr = block_resimulation_update(tr)
    end

    # start saving traces from here (for posterior restoration)
    trs = []
    for iter=1:n_samples
        tr = block_resimulation_update(tr)
        push!(trs, tr)
    end

    trs

end;

trs = block_resimulation_inference(n_count_data, alpha, count_data, 10000, 30000)

tau_samples = [trs[i][:tau] for i=1:30000]
lambda_1_samples = [trs[i][:lambda_1] for i=1:30000]
lambda_2_samples = [trs[i][:lambda_2] for i=1:30000]



# histogram of the samples:

hist(lambda_1_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label="posterior of \$\\lambda_1\$", color="#A60628")
xlabel("\$\\lambda_1\$ value")
plt[:show]()

hist(lambda_2_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label="posterior of \$\\lambda_2\$", color="#A60628")
xlabel("\$\\lambda_2\$ value")
plt[:show]()

hist(tau_samples, bins=n_count_data, alpha=1,
         label="posterior of \$\\tau\$",
         color="#467821")
xlabel("\$\\tau\$ (in days)")
plt[:show]()


# histogram of the samples:

ax = subplot(311)
ax.set_autoscaley_on(false)
hist(lambda_1_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label="posterior of \$\\lambda_1\$", color="#A60628", density=true)
legend(loc="upper left")
title("Posterior distributions of the variables \$\\lambda_1,\\;\\lambda_2,\\;\\tau\$")
xlabel("\$\\lambda_1\$ value")
xlim([15, 30])

ax = subplot(312)
ax.set_autoscaley_on(false)
hist(lambda_2_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label="posterior of \$\\lambda_2\$", color="#7A68A6", density=true)
legend(loc="upper left")
xlabel("\$\\lambda_2\$ value")
xlim([15, 30])

subplot(313)
N = length(tau_samples)
w = fill(1.0 / N, N)
hist(tau_samples, bins=n_count_data, alpha=1,
         label="posterior of \$\\tau\$",
         color="#467821", weights=w, rwidth=2.)
xticks(1:n_count_data)
legend(loc="upper left")
ylim([0, .75])
xlim([35,  n_count_data- 20])
xlabel("\$\\tau\$ (in days)")
ylabel("probability")

plt[:show]()

# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples from the corresponding posterior distribution
N = length(tau_samples)
expected_texts_per_day = zeros(n_count_data)
for day = 1:n_count_data

    # ix1 and ix2 are bool indexes of all tau samples corresponding to
    # the switchpoint occurring before or after of the value of 'day'
    ix1 = findall(x -> day < x, tau_samples)
    ix2 = findall(x -> day >= x, tau_samples)
    # Each posterior sample corresponds to a value for tau.
    # for each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    # As explained, the "message count" random variable is Poisson distributed,
    # and therefore lambda (the poisson parameter) is the expected value of
    # "message count".
    expected_texts_per_day[day] = (sum(lambda_1_samples[ix1])
                                   + sum(lambda_2_samples[ix2])) / N

end

plot(1:n_count_data, expected_texts_per_day, lw=4, color="#E24A33",
         label="expected number of text-messages received")
xlim(0, n_count_data)
xlabel("Day")
ylabel("Expected # text-messages")
title("Expected number of text-messages received")
ylim(0, 60)
bar(1:n_count_data, count_data, color="#348ABD", alpha=0.65,
        label="observed texts per day")

legend(loc="upper left")
plt[:show]()

println(mean([lambda_1_samples[i] < lambda_2_samples[i] for i=1:30000]))
