using Agents
using Random
import ImageMagick
using FileIO: load


@agent Smelt GridAgent{3} begin
energy::Float64
end




function initialize_model(
    dims2 = (100, 100, 50),
    heightmap_url =
    "https://raw.githubusercontent.com/JuliaDynamics/" *
    "JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png",  
    n_rabbits = 1,  ## initial number of rabbits
    dt = 0.1,   ## discrete timestep each iteration of the model
    seed = 42,  ## seed for random number generator
)

    # Download and load the heightmap. The grayscale value is converted to `Float64` and
    # scaled from 1 to 40
    heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
    # The x and y dimensions of the pathfinder are that of the heightmap
    #dims = (size(heightmap)..., 50)
    
    # The region of the map that is accessible to each type of animal (land-based or flying)
    # is defined using `BitArrays`
    #land_walkmap = BitArray(trues(dims...))


    # Generate the RNG for the model
    rng = MersenneTwister(seed)

    # Note that the dimensions of the space do not have to correspond to the dimensions
    # of the pathfinder. Discretisation is handled by the pathfinding methods
    #space = ContinuousSpace((100., 100., 50.); periodic = false)
    space = GridSpace(dims2, periodic = false)

    # Generate an array of random numbers, and threshold it by the probability of grass growing
    # at that location. Although this causes grass to grow below `water_level`, it is
    # effectively ignored by `land_walkmap`
   
    properties = (
        heightmap = heightmap,
        dt = dt,
    )

    model = ABM(Smelt, space; rng, properties)

    # spawn each animal at a random walkable position according to its pathfinder
    #for _ in 1:n_rabbits
    #    energy = 5.0
    #    add_agent!(Smelt, model, energy)
    #end
    return model
end  


function animal_step!(smelt::Smelt, model)
    near_cells = nearby_positions(smelt.pos, model, 1)
    walk!(smelt, sample(near_cells.itr.iter, 1)[1], model)
end








int_model = initialize_model() 
int_model



using InteractiveDynamics
using GLMakie # CairoMakie doesn't do 3D plots



function static_preplot!(ax, model)
    surface!(
        ax,
        (100/205):(100/205):100,
        (100/205):(100/205):100,
        model.heightmap;
        colormap = :terrain
    )
end


animalcolor(a) = :red

abmvideo(
    "rabbit_fox_hawk.mp4",
    int_model, animal_step!;
    figure = (resolution = (800, 700),),
    frames = 300,
    framerate = 15,
    ac = animalcolor,
    as = 1.0,
    static_preplot!,
    title = "Rabbit Fox Hawk with pathfinding"
)