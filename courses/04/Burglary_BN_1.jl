using Statistics
using Gen

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

