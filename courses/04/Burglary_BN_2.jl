using PyCall
using Statistics
using Gen

function p_A(B, E)

   if B && E
      p = 0.95
   elseif B && !E
      p = 0.94
   elseif !B && E
      p = 0.29
   else
      p = 0.001
   end

   p

end


function p_J(A)

   if A
      p = 0.9
   else
      p = 0.05
   end

   p

end

function p_M(A)

   if A
      p = 0.7
   else
      p = 0.01
   end

   p

end

@gen (static) function Burglary_Model()

   B ~ bernoulli(0.001)
   E ~ bernoulli(0.002)
   A ~ bernoulli(p_A(B,E))
   J ~ bernoulli(p_J(A))
   M ~ bernoulli(p_M(A))

end


Gen.@load_generated_functions

#@pyimport graphviz
#using Gen: draw_graph
#draw_graph(Burglary_Model, graphviz, "graph")

function make_constraints()
    constraints = Gen.choicemap()
    constraints[:J] = true
    constraints[:M] = true
    constraints
end;

function block_resimulation_update(tr)

    # Block 1: Update B
    latent_variable = select(:B)
    (tr, _) = mh(tr, latent_variable)

    # Block 2: Update E
    latent_variable = select(:E)
    (tr, _) = mh(tr, latent_variable)

    # Block 3: Update A
    latent_variable = select(:A)
    (tr, _) = mh(tr, latent_variable)

    tr

end

function block_resimulation_inference(n_burnin, n_samples)
    observations = make_constraints()
    (tr, _) = generate(Burglary_Model, (), observations)
    for iter=1:n_burnin
        tr = block_resimulation_update(tr)
    end
    trs = []
    for iter=1:n_samples
        tr = block_resimulation_update(tr)
        push!(trs, tr)
    end

    trs

end;

trs = block_resimulation_inference(100000, 1000000)

B_samples = [trs[i][:B] for i=1:1000000]

println(mean(B_samples))

