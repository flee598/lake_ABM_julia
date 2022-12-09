using Agents, Random
using Distributions
using Pkg


# geneic traits of all fish
@agent Smelt GridAgent{2} begin
    energy::Float64               # current energy level
    reproduction_prob::Float64    # prob of reproducing
    Δenergy::Float64              # energy from food
    length::Float64               # fish length
end

# geneic traits of all fish
@agent Trout GridAgent{2} begin
    energy::Float64               # current energy level
    reproduction_prob::Float64    # prob of reproducing
    Δenergy::Float64              # energy from food
    length::Float64               # fish length
end



mu = 0
sigma = 1





function initialize_model(;
    n_smelt = 100,
    n_trout = 50,
    dims = (20, 20),
    regrowth_time = 30,           # basal resource growth rate
    Δenergy_smelt = 4,
    Δenergy_trout = 20,
    smelt_reproduce = 0.04,
    trout_reproduce = 0.05,
    length_smelt_mean = 12,
    length_trout_mean = 20,
    seed = 23182,
)

rng = MersenneTwister(seed)
space = GridSpace(dims, periodic = true)

# define littoral/pelagic matrix (1 = littoral 0 = pelagic)
m1_add = 1
m1 = zeros(Int, dims)
m1[size(m1, 1) - m1_add:size(m1, 1), :] .= 1
m1[1:1 + m1_add, :] .= 1
m1[:, size(m1, 2) - m1_add:size(m1, 2)] .= 1
m1[:, 1:1 + m1_add] .= 1


# Model properties contain the basal resource as two arrays: whether it is fully grown
# and the time to regrow. Also have static parameter `regrowth_time`.
# Notice how the properties are a `NamedTuple` to ensure type stability.
properties = (
    basal_type = m1,
    fully_grown = falses(dims),
    countdown = zeros(Int, dims),
    regrowth_time = regrowth_time,
)

model = ABM(Union{Smelt, Trout}, space;
    properties, rng, scheduler = Schedulers.randomly, warn = false
)
# Add agents
for _ in 1:n_smelt
    energy = rand(model.rng, 1:(Δenergy_smelt*2)) - 1
    length_smelt = round(rand(Normal(length_smelt_mean, 1), 1)[1], digits = 3)
    add_agent!(Smelt, model, energy, smelt_reproduce, Δenergy_smelt, length_smelt)
end

for _ in 1:n_trout
    energy = rand(model.rng, 1:(Δenergy_trout*2)) - 1
    length_trout = round(rand(Normal(length_trout_mean, 1), 1)[1], digits = 3)
    add_agent!(Trout, model, energy, trout_reproduce, Δenergy_trout, length_trout)
end


# Add basal resource at random initial levels
for p in positions(model)
    fully_grown = rand(model.rng, Bool)
    countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
    model.countdown[p...] = countdown
    model.fully_grown[p...] = fully_grown
end

return model
end


sheepwolfgrass = initialize_model()

sheepwolfgrass.properties




sample(1:5, 4, replace=false)


function sheepwolf_step!(smelt::Smelt, model)
    
    # move agent - rand implements a random walk, agent will move ± 1 in any direction
    walk!(smelt, rand, model)

    smelt.energy -= 1
    if smelt.energy < 0
        kill_agent!(smelt, model)
        return
    end

    eat!(smelt, model)
    if rand(model.rng) ≤ smelt.reproduction_prob
        reproduce!(smelt, model)
    end
end



function sheepwolf_step!(smelt::Smelt, model)
    
    # move agent - rand implements a random walk, agent will move ± 1 in any direction
    walk!(smelt, rand, model)


    near_cells = nearby_positions(agent.pos, model, r = 1)
    
    near_cells[fully_grown == T]

    for cells in near_cells
        model.fully_grown

    smelt.energy -= 1
    if smelt.energy < 0
        kill_agent!(smelt, model)
        return
    end

    eat!(smelt, model)
    if rand(model.rng) ≤ smelt.reproduction_prob
        reproduce!(smelt, model)
    end
end
