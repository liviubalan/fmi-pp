using PyPlot
using Gen

@gen function triangular_model()

    x1 ~ uniform(0, 1)
    x2 ~ uniform(0, 1)

    z = abs(x1 - x2)

end

traces = [Gen.simulate(triangular_model, ()) for _=1:30000];

z_samples = [traces[i][] for i=1:30000]

hist(z_samples, bins = 40)
plt[:show]()

