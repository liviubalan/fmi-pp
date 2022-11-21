using PyPlot
using Distributions

tau = rand(0:80)
println(tau)

alpha = 20.
lambda_1 = rand(Exponential(alpha))
lambda_2 = rand(Exponential(alpha))

println(lambda_1, " ", lambda_2)

data = [rand(Poisson(lambda_1), tau); rand(Poisson(lambda_2), 80 - tau)]

bar(1:80, data, color="#348ABD")
bar(tau, data[tau], color="r", label="user behaviour changed")
xlabel("Time (days)")
ylabel("count of text-msgs received")
title("Artificial dataset")
xlim(0, 80)
legend()
plt[:show]()




