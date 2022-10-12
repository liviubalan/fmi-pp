using Statistics
using Gen

@gen function model()

   c = 10

   # X = c
   X ~ uniform(0, 4 * c)

   function loss()

      return c < X ? c / (c + X) : X / (c + X)
      
   end

   return loss()

end

traces = [Gen.simulate(model, ()) for _=1:30000];

print("Bayes Error: ")
println(mean([traces[i][] for i=1:30000]))
