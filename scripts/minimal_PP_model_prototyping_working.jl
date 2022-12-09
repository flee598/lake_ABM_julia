using Agents
using Random
using Distributions
using InteractiveDynamics
using GLMakie


# geneic traits of all fish
@agent Smelt GridAgent{2} begin
    energy::Float64               # current energy level
    reproduction_prob::Float64    # prob of reproducing
    Δenergy::Float64              # energy from food
    length::Float64               # fish length
end



function initialize_model(;
    n_smelt = 3,
    dims = (20, 20),
    regrowth_time = 30,           # basal resource growth rate
    Δenergy_smelt = 5,            # growth from eating
    smelt_reproduce = 0.04,
    length_smelt_mean = 12,
    seed = 23182,
    basal_resource_init = 20,     # initial amount of basal resource
)


rng = MersenneTwister(seed)
space = GridSpace(dims, periodic = false)


# Model properties
properties = (
    fully_grown = falses(dims),
    countdown = zeros(Int, dims),
    regrowth_time = regrowth_time,
    basal_resource = zeros(Float64, dims),
)

model = ABM(Smelt, space;
    properties, rng, scheduler = Schedulers.randomly, warn = false
)
# Add agents
for _ in 1:n_smelt
    energy = rand(model.rng, 1:(Δenergy_smelt*2)) - 1
    length_smelt = round(rand(Normal(length_smelt_mean, 1), 1)[1], digits = 3)
    add_agent!(Smelt, model, energy, smelt_reproduce, Δenergy_smelt, length_smelt)
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


function sheepwolf_step!(smelt::Smelt, model)
    #walk!(smelt, rand, model)
    near_cells = nearby_positions(smelt.pos, model, 1)

    # find which of the nearby cells have resources
    grassy_cells = [] # do I need to prefdine the type here?

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

    # reproduction - is tick is within spawning window 

    # if rand(model.rng) ≤ smelt.reproduction_prob
    #    reproduce!(smelt, model)
    #end

end

# smelt eating
function eat!(smelt::Smelt, model)
    if model.basal_resource[smelt.pos...] > 0       # if there are rsources available 
        smelt.energy += smelt.Δenergy               # give smelt energy
        smelt.length += smelt.Δenergy               # grow smelt
        model.basal_resource[smelt.pos...] -= 0.1   # and reduce resources
        # currently resources can go negative ... fix
    end
    return
end

# smlet reproduction and post reproduction mortality
#function reproduce_!(agent::A, model) where {A}
#    id = nextid(model)
#    length = 12
#    energy = 10
#    offspring = A(id, agent.pos, length, energy)
#    add_agent_pos!(offspring, model)
#
#    # adults die based on probability (95% chance die after spawning)
#    if rand(Uniform(0,0.1),1) > 0.05
#        kill_agent!(smelt, model)
#    end
#    return
#end

#function grass_step!(model)
#    model.tick[1] += 1
#end



# near_cells = nearby_positions(sheepwolfgrass[2], sheepwolfgrass, 1)
# near_cells
# xx = collect(near_cells)
# xx
#near_cells
# randomly select on of  the cells that contains grass - move agent there
#move_agent!(sheep, sample(cell_w_grass), model)

# access agent params
# sheepwolfgrass[1]
# sheepwolfgrass[1].pos

#offset(a) = a isa Smelt ? (-0.1, -0.1*rand()) : (+0.1, +0.1*rand())
#ashape(a) = a isa Smelt ? :circle : :utriangle
#acolor(a) = a isa Smelt ? RGBf(rand(3)...) : RGBAf(0.2, 0.2, 0.3, 0.8)

#grasscolor(model) = model.countdown ./ model.regrowth_time

#heatkwargs = (colormap = [:brown, :green], colorrange = (0, 1))

#plotkwargs = (;
#    ac = acolor,
#    as = 25,
#    am = ashape,
#    offset,
#    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
#    heatarray = grasscolor,
#    heatkwargs = heatkwargs,
#)

#sheepwolfgrass = initialize_model()

#fig, ax, abmobs = abmplot(sheepwolfgrass;
#    agent_step! = sheepwolf_step!,
#plotkwargs...)
#fig



# data collection 

sheepwolfgrass = initialize_model()
steps = 10

adata = [:pos, :energy, :length]
#mdata = [:tick]

#adf, mdf = run!(sheepwolfgrass, sheepwolf_step!, steps; adata, mdata)
adf = run!(sheepwolfgrass, sheepwolf_step!, steps; adata)

adf

adf