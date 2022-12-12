using Agents
using Random
using Distributions

# sheep traits
@agent Sheep GridAgent{2} begin
    energy::Float64
end

# set up model
function initialize_model(;
    n_sheep = 3,
    dims = (20, 20),
    seed = 1234,
    )
    rng = MersenneTwister(seed)
    space = GridSpace(dims, periodic = false)
    
    # Model properties
    properties = Dict(
        :fully_grown => falses(dims),
        :tick => 0,
        )



    model = ABM(Sheep, space;
        properties, rng, scheduler = Schedulers.randomly, warn = false
    )
    
    # Add agents
    for _ in 1:n_sheep
        energy = rand(model.rng, 1:5) - 1
        add_agent!(Sheep, model, energy)
    end
    
    # Add grass resource
    for p in positions(model)
        fully_grown = rand(model.rng, Bool)
        model.fully_grown[p...] = fully_grown
    end
    return model
end


# move sheep
function sheepwolf_step!(sheep::Sheep, model)
walk!(sheep, rand, model)
end

 # do something to grass
 function grass_step!(model)
    @inbounds for p in positions(model)
        model.fully_grown[p...] = rand(model.rng, Bool)     
    end
    # update model counter
    model.tick += 1
end


sheepwolfgrass = initialize_model()
steps = 5
adata = [:pos, :energy]
mdata = [:fully_grown, :tick]

#adf = run!(sheepwolfgrass, sheepwolf_step!, steps; adata)
adf, mdf = run!(sheepwolfgrass, sheepwolf_step!, grass_step!, steps; adata, mdata)
adf
mdf


parameters = Dict(
    :min_to_be_happy => 0, # expanded
    :numagents => zeros((5,5)),         # expanded
    :griddims => (20, 20),            # not Vector = not expanded
)