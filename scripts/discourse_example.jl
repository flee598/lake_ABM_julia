using Agents
using Random
using Distributions
using InteractiveDynamics
using GLMakie

# sheep traits
@agent Sheep GridAgent{2} begin
    energy::Float64
end

# set up model
function initialize_model(;
    n_sheep = 3,
    regrowth_time = 30,
    dims = (20, 20),
    seed = 1234,
    )

    rng = MersenneTwister(seed)
    space = GridSpace(dims, periodic = false)

    # added n_sheep here to try and get it work on an interactive slider
    properties = Dict(
        :n_sheep => n_sheep,
        :fully_grown => falses(dims),
        :countdown => zeros(Int, dims),
        :regrowth_time => regrowth_time,
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
        countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
        model.countdown[p...] = countdown
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
end


# plotting stuff
grasscolor(model) = model.countdown ./ model.regrowth_time
heatkwargs = (colormap = [:brown, :green], colorrange = (0, 1))

plotkwargs = (;
    ac = RGBAf(1.0, 1.0, 1.0, 0.8),
    as = 25,
    am = '✿',
    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
    heatarray = grasscolor,
    heatkwargs = heatkwargs,
)

sheepwolfgrass = initialize_model()

# parameter I want on a slider
params = Dict(
    :n_sheep => 1:20,
)


fig, ax, abmobs = abmplot(sheepwolfgrass;
    agent_step! = sheepwolf_step!,
    model_step! = grass_step!,
    params,
plotkwargs...)
fig

