using Agents
using Random
using Distributions
using InteractiveDynamics
using GLMakie

# define agents -----------------------------------------------------------------------------
# geneic traits of all fish
@agent Smelt GridAgent{2} begin
    energy::Float64               # current energy level
    Δenergy::Float64              # energy from food
    reproduction::Float64    # prob of reproducing
    length::Float64  
    pelagic_pref::Float64
end


# initialize model --------------------------------------------------------------------------
function initialize_model(;
    n_smelt = 3,
    dims = (20, 20),
    regrowth_time = 30,           # basal resource growth rate
    Δenergy_smelt = 5,            # growth from eating
    reproduction_smelt = 1.0,
    length_smelt_mean = 12,
    seed = 23182,
    basal_resource_init = 20,     # initial amount of basal resource
    pelagic_pref_smelt = 1,
)

rng = MersenneTwister(seed)
space = GridSpace(dims, periodic = false)

# define littoral/pelagic matrix (1 = pelagic 0 = littoral)
m1_add = 1
m1 = zeros(Int, dims)
m1[size(m1, 1) - m1_add:size(m1, 1), :] .= 1
m1[1:1 + m1_add, :] .= 1
m1[:, size(m1, 2) - m1_add:size(m1, 2)] .= 1
m1[:, 1:1 + m1_add] .= 1


# Model properties
properties = Dict(
    :fully_grown => falses(dims),
    :countdown => zeros(Int, dims),
    :regrowth_time => regrowth_time,
    :basal_resource => zeros(Float64, dims),
    :basal_resource_type => m1,
    :tick => 0::Int64,
)

model = ABM(Smelt, space;
    properties, rng, scheduler = Schedulers.randomly, warn = false
)

# Add agents
for _ in 1:n_smelt
    energy = rand(model.rng, 1:(Δenergy_smelt*2)) - 1
    length = round(rand(Normal(length_smelt_mean, 1), 1)[1], digits = 3)
    pelagic_pref = pelagic_pref_smelt
    reproduction = reproduction_smelt
    add_agent!(Smelt, model, energy, reproduction, Δenergy_smelt, length, pelagic_pref)
end

# Add basal resource at random initial levels
for p in positions(model)
    fully_grown = rand(model.rng, Bool)
    countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
    model.countdown[p...] = countdown
    model.fully_grown[p...] = fully_grown
    model.basal_resource[p...] = round(rand(Normal(basal_resource_init, 1), 1)[1], digits = 3)
end
return model
end


# define agent movement --------------------------------------------------------------------------
function sheepwolf_step!(smelt::Smelt, model)
    #walk!(smelt, rand, model)
    near_cells = nearby_positions(smelt.pos, model, 1)

    # find which of the nearby cells have resources
    grassy_cells = [] # do I need to predefine the type here?

    for cell in near_cells
        if model.fully_grown[cell...]
            push!(grassy_cells, cell)
        end
    end

    # randomly choose one of the nearby cells with resources
    move_agent!(smelt, sample(grassy_cells, 1)[1], model)

    # each time step smelt loose 0.5 unit of energy, if 0 die
    smelt.energy -= 0.5
    if smelt.energy < 0
        kill_agent!(smelt, model)
        return
    end

    # smelt eating - see function below
    eat!(smelt, model)

    # reproduction
    if model.tick == 4
        #if rand(model.rng) ≤ smelt.reproduction_prob
        reproduce_smelt!(smelt, model)
        #end
    end
end


# define agent eating -----------------------------------------------------------------------------
function eat!(smelt::Smelt, model)
    if model.basal_resource[smelt.pos...] > 0       # if there are rsources available 
        smelt.energy += smelt.Δenergy               # give smelt energy
        smelt.length += smelt.Δenergy               # grow smelt
        model.basal_resource[smelt.pos...] -= 0.1   # and reduce resources
        # currently resources can go negative ... fix
    end
    return
end

# define agent reproduction --------------------------------------------------------------------------
# smlet reproduction and post reproduction mortality
function reproduce_smelt!(agent::A, model) where {A}
    id = nextid(model)
    #length = 12
    #energy = 10
    #pelagic_pref = 1
    offspring = A(id, agent.pos, agent.energy, agent.Δenergy, agent.reproduction, agent.length, agent.pelagic_pref)

    # adding the agent  - should be added to pelagic
    add_agent_pos!(offspring, model)

    # adults die based on probability - high for testing
    if rand(model.rng) > 0.99
        kill_agent!(smelt, model)
    end
    return
end
# define model counter --------------------------------------------------------------------------------

function grass_step!(model)
    model.tick += 1
end



# define plotting vars -----------------------------------------------------------------------------
offset(a) = a isa Smelt ? (-0.1, -0.1*rand()) : (+0.1, +0.1*rand())
set(a) = a isa Smelt ? (-0.1, -0.1*rand()) : (+0.1, +0.1*rand())
ashape(a) = a isa Smelt ? :circle : :utriangle
acolor(a) = a isa Smelt ? RGBf(rand(3)...) : RGBAf(0.2, 0.2, 0.3, 0.8)

grasscolor(model) = model.countdown ./ model.regrowth_time

heatkwargs = (colormap = [:brown, :green], colorrange = (0, 1))

plotkwargs = (;
    ac = acolor,
    as = 25,
    am = ashape,
    offset,
    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
    heatarray = grasscolor,
    heatkwargs = heatkwargs,
)


# initialize model -----------------------------------------------------------------------------------

sheepwolfgrass = initialize_model()

fig, ax, abmobs = abmplot(sheepwolfgrass;
    agent_step! = sheepwolf_step!,
    model_step! = grass_step!,
plotkwargs...)
fig


rand(Uniform(0, 0.1), 1)

# data collection 

sheepwolfgrass = initialize_model()
steps = 5
adata = [:pos, :energy, :Δenergy, :length, :pelagic_pref]
mdata = [:basal_resource_type, :tick]

adf, mdf = run!(sheepwolfgrass, sheepwolf_step!, grass_step!, steps; adata, mdata)


adf
mdf
