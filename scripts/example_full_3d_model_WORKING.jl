using Agents, Agents.Pathfinding
using Random
import ImageMagick
using FileIO: load
using InteractiveDynamics
using GLMakie # CairoMakie doesn't do 3D plots



@agent Animal ContinuousAgent{3} begin
    type::Symbol # one of :rabbit, :fox or :hawk
    energy::Float64
end

const v0 = (0.0, 0.0, 0.0) # we don't use the velocity field here
Rabbit(id, pos, energy) = Animal(id, pos, v0, :rabbit, energy)
eunorm(vec) = √sum(vec .^ 2)


function initialize_model(
    heightmap_url =
    "https://raw.githubusercontent.com/JuliaDynamics/" *
    "JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png",
    water_level = 8,
    grass_level = 20,
    mountain_level = 35;
    n_rabbits = 160,  ## initial number of rabbits
    Δe_grass = 25,  ## energy gained from eating grass
    Δe_rabbit = 30,  ## energy gained from eating one rabbit
    rabbit_repr = 0.06,  ## probability for a rabbit to (asexually) reproduce at any step
    rabbit_vision = 6,  ## how far rabbits can see grass and spot predators
    rabbit_speed = 1.3, ## movement speed of rabbits
    regrowth_chance = 0.03,  ## probability that a patch of grass regrows at any step
    dt = 0.1,   ## discrete timestep each iteration of the model
    seed = 42,  ## seed for random number generator
)

    # Download and load the heightmap. The grayscale value is converted to `Float64` and
    # scaled from 1 to 40
    heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
    # The x and y dimensions of the pathfinder are that of the heightmap
    dims = (size(heightmap)..., 50)
    # The region of the map that is accessible to each type of animal (land-based or flying)
    # is defined using `BitArrays`
    land_walkmap = BitArray(falses(dims...))
    air_walkmap = BitArray(falses(dims...))
    for i in 1:dims[1], j in 1:dims[2]
        # land animals can only walk on top of the terrain between water_level and grass_level
        if water_level < heightmap[i, j] < grass_level
            land_walkmap[i, j, heightmap[i, j]+1] = true
        end
        # air animals can fly at any height upto mountain_level
        if heightmap[i, j] < mountain_level
            air_walkmap[i, j, (heightmap[i, j]+1):mountain_level] .= true
        end
    end

    # Generate the RNG for the model
    rng = MersenneTwister(seed)

    # Note that the dimensions of the space do not have to correspond to the dimensions
    # of the pathfinder. Discretisation is handled by the pathfinding methods
    space = ContinuousSpace((100., 100., 50.); periodic = false)

    # Generate an array of random numbers, and threshold it by the probability of grass growing
    # at that location. Although this causes grass to grow below `water_level`, it is
    # effectively ignored by `land_walkmap`
    grass = BitArray(
        rand(rng, dims[1:2]...) .< ((grass_level .- heightmap) ./ (grass_level - water_level)),
    )
    properties = (
        # The pathfinder for rabbits and foxes
        landfinder = AStar(space; walkmap = land_walkmap),
        # The pathfinder for hawks
        airfinder = AStar(space; walkmap = air_walkmap, cost_metric = MaxDistance{3}()),
        Δe_grass = Δe_grass,
        Δe_rabbit = Δe_rabbit,
        rabbit_repr = rabbit_repr,
        rabbit_vision = rabbit_vision,
        rabbit_speed = rabbit_speed,
        heightmap = heightmap,
        grass = grass,
        regrowth_chance = regrowth_chance,
        water_level = water_level,
        grass_level = grass_level,
        dt = dt,
    )

    model = ABM(Animal, space; rng, properties)

    # spawn each animal at a random walkable position according to its pathfinder
    for _ in 1:n_rabbits
        add_agent_pos!(
            Rabbit(
                nextid(model), ## Using `nextid` prevents us from having to manually keep track
                               # of animal IDs
                random_walkable(model, model.landfinder),
                rand(model.rng, Δe_grass:2Δe_grass),
            ),
            model,
        )
    end
   
    return model
end

function animal_step!(animal, model)
        rabbit_step!(animal, model)
   end


function rabbit_step!(rabbit, model)
    # Eat grass at this position, if any
    if get_spatial_property(rabbit.pos, model.grass, model) == 1
        model.grass[get_spatial_index(rabbit.pos, model.grass, model)] = 0
        rabbit.energy += model.Δe_grass
    end

    # The energy cost at each step corresponds to the amount of time that has passed
    # since the last step
    rabbit.energy -= model.dt
    # All animals die if their energy reaches 0
    if rabbit.energy <= 0
        kill_agent!(rabbit, model, model.landfinder)
        return
    end

    # Get a list of positions of all nearby predators
    predators = [
        x.pos for x in nearby_agents(rabbit, model, model.rabbit_vision) if
            x.type == :fox || x.type == :hawk
            ]
    
    # If the rabbit sees a predator and isn't already moving somewhere
    if !isempty(predators) && is_stationary(rabbit, model.landfinder)
        # Try and get an ideal direction away from predators
        direction = (0., 0., 0.)
        for predator in predators
            # Get the direction away from the predator
            away_direction = (rabbit.pos .- predator)
            # In case there is already a predator at our location, moving anywhere is
            # moving away from it, so it doesn't contribute to `direction`
            all(away_direction .≈ 0.) && continue
            # Add this to the overall direction, scaling inversely with distance.
            # As a result, closer predators contribute more to the direction to move in
            direction = direction .+ away_direction ./ eunorm(away_direction) ^ 2
        end
        # If the only predator is right on top of the rabbit
        if all(direction .≈ 0.)
            # Move anywhere
            chosen_position = random_walkable(rabbit.pos, model, model.landfinder, model.rabbit_vision)
        else
            # Normalize the resultant direction, and get the ideal position to move it
            direction = direction ./ eunorm(direction)
            # Move to a random position in the general direction of away from predators
            position = rabbit.pos .+ direction .* (model.rabbit_vision / 2.)
            chosen_position = random_walkable(position, model, model.landfinder, model.rabbit_vision / 2.)
        end
        plan_route!(rabbit, chosen_position, model.landfinder)
    end

    # Reproduce with a random probability, scaling according to the time passed each
    # step
    # rand(model.rng) <= model.rabbit_repr * model.dt && reproduce!(rabbit, model)
    rand(model.rng) <= 0.001



    # If the rabbit isn't already moving somewhere, move to a random spot
    if is_stationary(rabbit, model.landfinder)
        plan_route!(
            rabbit,
            random_walkable(rabbit.pos, model, model.landfinder, model.rabbit_vision),
            model.landfinder
        )
    end

    # Move along the route planned above
    move_along_route!(rabbit, model, model.landfinder, model.rabbit_speed, model.dt)
end

function reproduce!(animal, model)
    animal.energy = ceil(Int, animal.energy / 2)
    add_agent_pos!(Animal(nextid(model), animal.pos, animal.type, animal.energy), model)
end

function model_step!(model)
    # To prevent copying of data, obtain a view of the part of the grass matrix that
    # doesn't have any grass, and grass can grow there
    growable = view(
        model.grass,
        model.grass .== 0 .& model.water_level .< model.heightmap .<= model.grass_level,
    )
    # Grass regrows with a random probability, scaling with the amount of time passing
    # each step of the model
    growable .= rand(model.rng, length(growable)) .< model.regrowth_chance * model.dt
end


animalcolor(a) =
    if a.type == :rabbit
        :brown
    elseif a.type == :fox
        :orange
    else
        :blue
    end


function static_preplot!(ax, model)
        surface!(
            ax,
            (100/205):(100/205):100,
            (100/205):(100/205):100,
            model.heightmap;
            colormap = :terrain
        )
end


model = initialize_model()


abmvideo(
    "rabbit_fox_hawk.mp4",
    model, animal_step!, model_step!;
    figure = (resolution = (800, 700),),
    frames = 75,
    framerate = 15,
    ac = animalcolor,
    as = 1.0,
    static_preplot!,
    title = "Rabbit Fox Hawk with pathfinding"
) 




heightmap_url =
"https://raw.githubusercontent.com/JuliaDynamics/" *
"JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png"

heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
heightmap


dims = (size(heightmap)..., 50)
dims