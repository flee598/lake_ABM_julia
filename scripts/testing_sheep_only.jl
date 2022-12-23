using Agents, Random

@agent Sheep GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64
end


function initialize_model(;
    n_sheep = 100,
    dims = (20, 20),
    regrowth_time = 30,
    Δenergy_sheep = 4,
    sheep_reproduce = 0.04,
    seed = 23182,
)

rng = MersenneTwister(seed)
space = GridSpace(dims, periodic = true)
# Model properties contain the grass as two arrays: whether it is fully grown
# and the time to regrow. Also have static parameter `regrowth_time`.
# Notice how the properties are a `NamedTuple` to ensure type stability.
properties = (
    fully_grown = falses(dims),
    countdown = zeros(Int, dims),
    regrowth_time = regrowth_time,
)
model = ABM(Sheep, space;
    properties, rng, scheduler = Schedulers.randomly, warn = false
)

# Add agents
for _ in 1:n_sheep
    energy = rand(model.rng, 1:(Δenergy_sheep*2)) - 1
    add_agent!(Sheep, model, energy, sheep_reproduce, Δenergy_sheep)
end


# Add grass with random initial growth
for p in positions(model)
    fully_grown = rand(model.rng, Bool)
    countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
    model.countdown[p...] = countdown
    model.fully_grown[p...] = fully_grown
end
return model
end



function sheep_step!(sheep::Sheep, model)
    walk!(sheep, rand, model)
    sheep.energy -= 1
    if sheep.energy < 0
        kill_agent!(sheep, model)
        return
    end
    eat!(sheep, model)
    if rand(model.rng) ≤ sheep.reproduction_prob
        reproduce!(sheep, model)
    end
end

function eat!(sheep::Sheep, model)
    if model.fully_grown[sheep.pos...]
        sheep.energy += sheep.Δenergy
        model.fully_grown[sheep.pos...] = false
    end
    return
end

function reproduce!(agent::A, model) where {A}
    agent.energy /= 2
    id = nextid(model)
    offspring = A(id, agent.pos, agent.energy, agent.reproduction_prob, agent.Δenergy)
    add_agent_pos!(offspring, model)
    return
end

function grass_step!(model)
    @inbounds for p in positions(model) # we don't have to enable bound checking
        if !(model.fully_grown[p...])
            if model.countdown[p...] ≤ 0
                model.fully_grown[p...] = true
                model.countdown[p...] = model.regrowth_time
            else
                model.countdown[p...] -= 1
            end
        end
    end
end

sheepgrass = initialize_model()
steps = 100
adata = []
mdata = [:countdown]

# obtainer = copy - use this if you need to update the mdf output - by default if the output is mutable container it 
# won't show updates. using obtainer = copy will reduce performance, only use for prototyping 
adf, mdf = run!(sheepgrass, sheep_step!, grass_step!, steps; adata, mdata, obtainer = copy)


mdf[:,2][4] == mdf[:,2][3]
mdf


mdf[:,2][2]