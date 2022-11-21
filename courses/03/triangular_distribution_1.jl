using PyPlot
using Gen

include("Triangular.jl")

@gen function triangular_model()

    z = triangular(-3, 8, 0)

end

traces = [Gen.simulate(triangular_model, ()) for _=1:30000];

z_samples = [traces[i][] for i=1:30000]

hist(z_samples, bins = 40)
plt[:show]()
