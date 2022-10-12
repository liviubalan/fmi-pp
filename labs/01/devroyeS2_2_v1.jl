using Statistics
using Gen

@gen function model()

   c = 10

   # X = c
   X ~ uniform(0, 4 * c)
   Y ~ bernoulli(X / (c + X))

   function Yh()

      return X / (c + X) > 0.5 ? 1 : 0
      
   end

   return Yh()

end

traces = [Gen.simulate(model, ()) for _=1:30000];

print("Bayes Error: ")
println(mean([traces[i][:Y] != traces[i][] for i=1:30000]))
