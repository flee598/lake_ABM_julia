using Agents
using Random
using Distributions
using InteractiveDynamics
using GLMakie


# geneic traits of all fish
@agent Smelt GridAgent{2} begin
    energy::Float64               # current energy level
end


function initialize_model(;
    n_smelt = 3,
    dims = (20, 20),
    seed = 1234,
)
rng = MersenneTwister(seed)
space = GridSpace(dims, periodic = false)

# Model properties
properties = (
    fully_grown = falses(dims),
    tick =  Int
)

model = ABM(Smelt, space;
    properties, rng, scheduler = Schedulers.randomly, warn = false
)

# Add agents
for _ in 1:n_smelt
    energy = rand(model.rng, 1:5) - 1
    add_agent!(Smelt, model, energy)
end

# Add basal resource at random initial levels
for p in positions(model)
    fully_grown = rand(model.rng, Bool)
    model.fully_grown[p...] = fully_grown
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
end



function grass_step!(model)
    model.tick .+=  Int64[1]
end



sheepwolfgrass = initialize_model()
steps = 5
adata = [:pos, :energy]
mdata = [:tick]
adf, mdf = run!(sheepwolfgrass, sheepwolf_step!, grass_step!, steps; adata, mdata)
adf
mdf


mdf[:,2][2]

typeof(sheepwolfgrass.tick)

sheepwolfgrass.tick



xx
Array{Float64, 1}
Int

Int[1]


modeltick = Int64[1]

for i in 1:5
    modeltick += Int64[1]
end

modeltick