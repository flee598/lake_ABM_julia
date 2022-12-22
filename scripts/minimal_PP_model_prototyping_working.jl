using Agents
using Random
using Distributions
using InteractiveDynamics
using GLMakie

# define agents -----------------------------------------------------------------------------
# geneic traits of all fish
@agent Smelt GridAgent{2} begin
    energy::Float64                   # current energy level
    Δenergy::Float64                  # energy from food
    reproduction::Float64             # prob of reproducing
    consume_amount::Float64           # amount of resource consumed in one predation event
    length::Float64                   # length in mm
    vision_range::Int64               # number of cells agent can "see"
    mortality_random::Float64         # probability of random mortality / time step
    mortality_reproduction::Float64   # probability of mortality after reproduction
    resource_pref_adult::Float64      # smelt/koaro: 1 = pelagic, 2 = littoral, trout: 1 = smelt, 2 = koaro
    resource_pref_juv::Float64        # smelt/koaro: 1 = pelagic, 2 = littoral, trout: 1 = smelt, 2 = koaro
    stage::Int64                      # 0 = juvenile, 1 = adult
    growth_rate::Float64              # growth rate
    size_mature::Float64              # size individual transitions from juv to adult
    fecundity_mean::Int64             # mean number larvae produced
    fecundity_sd::Int64               # SD number larvae produced
end

# initialize model --------------------------------------------------------------------------
# note - starting out with just smelt/pelagic, will need variables below for koaro and trout and littoral

function initialize_model(;
    dims = (20, 20),                  # grid size
    cell_resource_growth_lit = 20,    # cell resource growth rate - littoral resources - draw from distribution?
    cell_resource_growth_pel = 20,    # cell resource growth rate - pelagic resources - draw from distribution?
    cell_resource_k_lit = 100,        # cell resource carrying capacity - littoral - draw from distribution?
    cell_resource_k_pel = 100,        # cell resource carrying capacity - littoral - draw from distribution?
    n_smelt = 3,                      # initial number of smelt
    Δenergy_smelt = 5.0,              # energy gained from eating 1 unit resource - draw from distribution?
    consume_amount_smelt = 5.0,       # max amount consumed in 1 timestep - draw from distribution?
    breed_prob_smelt = 0.9,           # probability of spawning (during seasonal window only) - draw from distribution?
    breed_mortality_smelt = 0.5,      # probability of dying after spawning - draw from distribution?
    growth_rate_smelt = 1.0,          # mm growth / time step - draw from distribution?
    length_mean_smelt = 15.0,         # mean adult smelt length - used for setting initial lengths
    length_sd_smelt = 1.0,            # SD adult smelt length - used for setting initial lengths
    vision_smelt = 1,                 # number of cells smelt can "see"
    n_juv_mean_smelt = 100,          # mean number of juveniles produced by 1 female (note this is juveniles, not eggs)
    n_juv_sd_smelt = 10,             # SD number of juveniles produced by 1 female (note this is juveniles, not eggs)
    size_maturity_smelt = 10.0,         # size (mm) when smelt transition from juvenile to adult
    mortality_random_smelt = 0.01,    # probability of random mortality each timestep - draw from distribution?
    resource_pref_adult_smelt = 1.0,  # adult smelt preference for (1) pelagic (0) littoral resources - but if koaro larvae present consume them
    resource_pref_juv_smelt = 1.0,    # juvenile preference for pelagic (1) or littoral (0) resources
    stage = 1,
    seed = 23182,
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
    :basal_resource => zeros(Float64, dims),
    :basal_resource_type => m1,
    :cell_resource_growth_lit => cell_resource_growth_lit::Int64,
    :cell_resource_growth_pel => cell_resource_growth_pel::Int64,
    :cell_resource_k_lit => cell_resource_k_lit::Int64,
    :cell_resource_k_pel => cell_resource_k_pel::Int64,
    :tick => 0::Int64,
    )

model = ABM(Smelt, space;
    properties, rng, scheduler = Schedulers.randomly, warn = false
)

# Add agents - need to figure out how to add different types of agents
for _ in 1:n_smelt
    energy = rand(model.rng, 1:(Δenergy_smelt*2)) - 1
    Δenergy = Δenergy_smelt
    reproduction = breed_prob_smelt
    consume_amount = consume_amount_smelt
    length = round(rand(Normal(length_mean_smelt, length_sd_smelt), 1)[1], digits = 3)
    vision_range = vision_smelt
    mortality_random = mortality_random_smelt
    mortality_reproduction = breed_mortality_smelt
    resource_pref_adult = resource_pref_adult_smelt
    resource_pref_juv = resource_pref_juv_smelt
    stage = stage
    growth_rate = growth_rate_smelt
    size_mature = size_maturity_smelt
    fecundity_mean = n_juv_mean_smelt
    fecundity_sd = n_juv_sd_smelt

    # params here must be in the same order as the @agent section above
    add_agent!(Smelt, model, energy, Δenergy, reproduction, consume_amount, length, vision_range, mortality_random,
               mortality_reproduction, resource_pref_adult, resource_pref_juv, stage, growth_rate, size_mature, fecundity_mean, fecundity_sd)
end


# Add basal resource at random initial levels
for p in positions(model)
    fully_grown = rand(model.rng, Bool)
    countdown = fully_grown ? cell_resource_growth_pel : rand(model.rng, 1:cell_resource_growth_pel) - 1
    model.countdown[p...] = countdown
    model.fully_grown[p...] = fully_grown
    model.basal_resource[p...] = rand(model.rng, 5:cell_resource_k_pel) - 1
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
    length = 12
    energy = 10.0
    Δenergy = agent.Δenergy/2
    stage = 0
    offspring = A(id, agent.pos, energy, Δenergy, agent.reproduction, agent.consume_amount, length,
    agent.vision_range, agent.mortality_random, agent.mortality_reproduction, agent.resource_pref_adult,
    agent.resource_pref_juv, stage, agent.growth_rate, agent.size_mature, agent.fecundity_mean, agent.fecundity_sd)


    # adding the agent  - should be added to pelagic
    add_agent_pos!(offspring, model)

    # adults die based on probability - high for testing - will this kill all agents or is it run run for each agent?
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

grasscolor(model) = model.countdown

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

# data collection 

sheepwolfgrass = initialize_model()
steps = 5
adata = [:pos, :energy, :Δenergy, :reproduction, :consume_amount, :length, :vision_range, :mortality_random,
:mortality_reproduction, :resource_pref_adult, :resource_pref_juv, :stage, :growth_rate, :size_mature, :fecundity_mean, :fecundity_sd]
mdata = [:basal_resource_type, :tick]

adf, mdf = run!(sheepwolfgrass, sheepwolf_step!, grass_step!, steps; adata, mdata)


adf
mdf
