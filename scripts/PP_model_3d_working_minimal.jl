#=
3D model of a lake food web.

Overview:
1: generate a 3D lake environemnt from bathymetry data (currently csv)
2: populate lake with fish - fish have a variety of traits (e.g. length)
3: define lake cell traits (e.g. resource type and level)
4: define fish movement (random in 3D, but weighted to move towards resources)
5: define lake resource step (e.g. resource growth)
6: repeat for n model steps
=#

#= 
To do
add multiple agents
add resources
add agent behaviour 

=#


# load packages ---------------------------------------------------------------------
using Agents, Agents.Pathfinding
using Random
using StatsBase
using CSV                   # importing bathymetry from csv
using DataFrames            # importing bathymetry from csv
using InteractiveDynamics   # interactive plots
using GLMakie               # interactive plots
# using FileIO: load        # used to download example heightmap

# define agents
@agent Fish GridAgent{3} begin
end

# initialize model ------------------------------------------------------------------------
function initialize_model(;
    #heightmap_url =
    #"https://raw.githubusercontent.com/JuliaDynamics/" *
    #"JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png", 
    lake_url = "data\\taupo_500m.csv",
    n_fish = 10,
    max_littoral_depth = 50,    # how far down does the littoral go in meters
    seed = 12345,
)


# try removing lake walls so we can see what is going on in the videos - doesn't work as the littoral zone is much shallower
# than the pelagic so it obscures the view
# heightmap2 = replace(heightmap2, 10155 => 0)



# example topology
#heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1

# lake_url = "data\\taupo_500m.csv"

# load lake topology ----------------------------------------------------------
lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix

# convert to integer
heightmap = floor.(Int, convert.(Float64, lake_mtx))


# rescale so that there aren't negative depth values (the deepest point in the lake = 1)
heightmap2 = heightmap .+ abs(minimum(heightmap)) .+ 1
heightmap2 .= heightmap .*-1


# try removing lake walls so we can see what is going on in the videos - doesn't work as the littoral zone is much shallower
# than the pelagic so it obscures the view
#heightmap2 = replace(heightmap2, -9999 => 0)

# lake depth 
mx_dpth = maximum(heightmap2)

# create new lake_type variable -----------------------------------------------
# 1 = littoral, 0 = pelagic
lake_type = ones(Int, size(heightmap2))
lake_type .= heightmap2



# define the maximum depth of the littoral zone - this could be automated epending on env conditions


# if take_type (depth) is between 0 and max_littoral_depth cell is littoral
lake_type[lake_type .> 0 .&& lake_type .< (max_littoral_depth + 1)] .= 1

# if lake is deeper than max_littoral_depth cell is pelagic
lake_type[lake_type .> max_littoral_depth] .= 0

# set limits between which agents can exist (surface and lake bed) ----------------
# currently agents can get out of the water a bit!
lake_surface_level = mx_dpth
lake_floor = 1


# lake dimensions ---------------------------------------------------------
dims = (size(heightmap2)..., mx_dpth)


# Note that the dimensions of the space do not have to correspond to the dimensions of the heightmap ... dont understand how this works... 
# might only be for continuous spaces
#space = GridSpace((100, 100, 50), periodic = false)
space = GridSpace(dims, periodic = false)

#  swimable space
swim_walkmap = BitArray(falses(dims...))

 # fish can swim at any between the lake bed and lake suface
for i in 1:dims[1], j in 1:dims[2]
    if lake_floor < heightmap2[i, j] < lake_surface_level
        swim_walkmap[i, j, (heightmap2[i, j]+1):lake_surface_level] .= true
    end
end


# create lake basal resource array
lake_basal_resource = zeros(size(heightmap2))


# model properties
properties = (
    swim_walkmap = swim_walkmap,
    waterfinder = AStar(space; walkmap = swim_walkmap, diagonal_movement = true),
    heightmap2 = heightmap2,
    lake_type = lake_type,
    lake_basal_resource = lake_basal_resource,
    fully_grown = falses(dims),
)

# rng
rng = MersenneTwister(seed)

model = ABM(Fish, space; properties, rng, scheduler = Schedulers.randomly, warn = false
)

# Add agents
for _ in 1:n_fish
    add_agent_pos!(
        Fish(
            nextid(model), 
            random_walkable(model, model.waterfinder),
        ),
        model,
    )
end

# here is where I add cell "traits" such as resource level
for p in positions(model)
    fully_grown = rand(model.rng, Bool)
    model.fully_grown[p...] = fully_grown
end

return model

end

# fish movement - random ----------------------------------------------------
function fish_step!(fish::Fish, model)
    
    # 1 = vison range
    near_cells = nearby_positions(fish.pos, model, 1)
    
    # storage
    grassy_cells = []
    
    # find which of the nearby cells are allowed to be moved onto
    for cell in near_cells
        if model.swim_walkmap[cell...] > 0
            push!(grassy_cells, cell)
        end
    end
    
    if length(grassy_cells) > 0
        # sample 1 cell
        m_to = sample(grassy_cells)
        # move
        move_agent!(fish, m_to, model)
    end
end

# resource step - something mundane - TO BE UPDATED -------------------------------------
function lake_resource_step!(model)
    for p in positions(model)
        model.fully_grown[p...] = true
    end
end


# set up model -----------------------------------------------------------------------------
model_initilised = initialize_model() 

# plotting params -------------------------------------------------------------------------
plotkwargs = (;
    ac = :red,
    as = 2,
    am = :circle,
    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
)


# interactive plot ------------------------------------------------------------------------
fig, ax, abmobs = abmplot(model_initilised;
    agent_step! = fish_step!,
    model_step! = lake_resource_step!,
plotkwargs...)
fig



# 3d video with bathymetry surface added ---------------------------------------------
function static_preplot!(ax, model)
    surface!(
        ax,
        #(100/205):(100/205):100,
        #(100/205):(100/205):100,
        model.heightmap2;
        colormap = :viridis
    )
end

#animalcolor(a) = :red

abmvideo(
    "rabbit_fox_hawk.mp4",
    model_initilised, fish_step!, lake_resource_step!;
    figure = (resolution = (1920 , 1080),),
    frames = 40,
    ac = :red,
    as = 1,
    static_preplot!,
    title = "Rabbit Fox Hawk with pathfinding"
)





lake_basal_resource = zeros(dims)


lake_basal_resource[]
lake_type


dims


for i in 1:dims[1], j in 1:dims[2]
    if xx[i,j]
        lake_basal_resource[i, j, 1:mx_dpth] .= 5.0
    end
end




lake_basal_resource
