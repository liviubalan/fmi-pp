using PyPlot
using Gen

include("Triangular.jl")

@gen function triangular_model()

    z = triangular(-3, 8, 0)

end

@gen function wrapper_model()

    z = @trace(triangular_model(), :z)

end

traces = [Gen.simulate(wrapper_model, ()) for _=1:30000];

z_samples = [traces[i][:z] for i=1:30000]

hist(z_samples, bins = 40)
plt[:show]()

