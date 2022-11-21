using Printf
using PyPlot
using Distributions

alpha = 20.
lambda = rand(Exponential(alpha))

data = rand(Poisson(lambda), 80)

open("txtdata_sim.csv", "w") do f
    for x in data
        @printf(f, "%e\n", x)
    end
end

bar(1:80, data, color="#348ABD")
xlabel("Time (days)")
ylabel("count of text-msgs received")
title("Artificial dataset")
xlim(0, 80)
plt[:show]()
