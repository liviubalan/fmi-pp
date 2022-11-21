using Statistics
using Gen
using GenVariableElimination

@gen function Burglary_Model()

   B ~ bernoulli(0.001)
   E ~ bernoulli(0.002)

   if B && E
      p_A = 0.95
   elseif B && !E
      p_A = 0.94
   elseif !B && E
      p_A = 0.29
   else
      p_A = 0.001
   end

   A ~ bernoulli(p_A)

   if A
      p_J = 0.9
   else
      p_J = 0.05
   end

   J ~ bernoulli(p_J)

   if A
      p_M = 0.7
   else
      p_M = 0.01
   end

   M ~ bernoulli(p_M)

end

function make_constraints()
    constraints = Gen.choicemap()
    constraints[:J] = true
    constraints[:M] = true
    constraints
end;

#trace = simulate(Burglary_Model, ())
(trace, _) = generate(Burglary_Model, (), make_constraints())

latents = Dict{Any,Latent}()
latents[:B] = Latent([true, false], [])
latents[:E] = Latent([true, false], [])
latents[:A] = Latent([true, false], [:B, :E])
latents[:J] = Latent([true, false], [:A])
latents[:M] = Latent([true, false], [:A])

observations = Dict{Any,Observation}()
observations[:J] = Observation([:J])
observations[:M] = Observation([:M])

factor_graph = compile_trace_to_factor_graph(trace, latents, observations)

using PyCall
@pyimport graphviz
draw_factor_graph(factor_graph, graphviz, "graph2", String)

elimination_order = Any[]
push!(elimination_order, :B)
push!(elimination_order, :E)
push!(elimination_order, :A)
push!(elimination_order, :J)
push!(elimination_order, :M)


elimination_result = variable_elimination(factor_graph, elimination_order)

println(elimination_result)
