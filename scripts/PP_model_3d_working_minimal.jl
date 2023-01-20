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
    #lake_surface_level = 200,
    max_littoral_depth = 20,    # how far down does the littoral go in meters
    seed = 23182,
)

# example topology
#heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1

# load lake topology ----------------------------------------------------------
lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix

# convert to integer
heightmap = floor.(Int, convert.(Float64, lake_mtx))

# rescale so that there aren't negative depth values (the depest point in the lake = 1)
heightmap2 = heightmap .+ abs(minimum(heightmap)) .+ 1

# try removing lake walls so we can see what is going on in the videos - doesn't work as the littoral zone is much shallower
# than the pelagic so it obscures the view
heightmap2 = replace(heightmap2, 10155 => 0)

# lake depth + a bit of buffer - AUTOMATE
mx_dpth = 200
#mx_dpth = abs(minimum(heightmap2)) + 10


# create new lake_type variable -----------------------------------------------
lake_type .= heightmap

# if take_type is between 0 and max_littoral_depth cell is littoral
lake_type[lake_type .< 1 .&& lake_type .> -(max_littoral_depth + 1)] .= 1

# if lake is deeper than max_littoral_depth cell is pelagic
lake_type[lake_type .< -max_littoral_depth] .= 0


# set limits between which agents can exist (surface and lake bed) ----------------
# currently agents can get out of the water a bit!
lake_surface_level = mx_dpth - 10
lake_floor = 1

# lake dimensions ---------------------------------------------------------
dims = (size(heightmap2)..., mx_dpth)


# Note that the dimensions of the space do not have to correspond to the dimensions of the heightmap ... dont understand how this works... 
# might only be for continuous spaces
#space = GridSpace((100, 100, 50), periodic = false)
space = GridSpace(dims, periodic = false)

# 
swim_walkmap = BitArray(falses(dims...))

 # fish can swim at any between the lake bed and lake suface
for i in 1:dims[1], j in 1:dims[2]
    if lake_floor < heightmap2[i, j] < lake_surface_level
        swim_walkmap[i, j, (heightmap2[i, j]+1):lake_surface_level] .= true
    end
end

# model properties
properties = (
    swim_walkmap = swim_walkmap,
    waterfinder = AStar(space; walkmap = swim_walkmap, diagonal_movement = true),
    heightmap2 = heightmap2,
    lake_type = lake_type,
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







typeof(NaN)

x = [1, missing]
x

typeof(x)

lake_url = "data\\taupo_500m.csv"



# load lake topology
lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix

# convert to integer
heightmap = floor.(Int, convert.(Float64, lake_mtx))


# create a littoral/pelagic layer








# testing bits of code -----------------------------------------------------------------
#=
heightmap_url =
    "https://raw.githubusercontent.com/JuliaDynamics/" *
    "JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png" 

heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
dims = (size(heightmap)..., 50)

swim_walkmap = BitArray(falses(dims...))

lake_surface_level = 35

for i in 1:dims[1], j in 1:dims[2]
    if heightmap[i, j] < lake_surface_level
        swim_walkmap[i, j, (heightmap[i, j]+1):lake_surface_level] .= true
    end
end


heightmap
swim_walkmap[:, :, 20]


near_cells = nearby_positions((1,1,2), model_initilised, 1)
near_cells


# storage
grassy_cells = []
 
# find which of the nearby cells are allowed to be moved onto
for cell in near_cells
    if model_initilised.swim_walkmap[cell...] > 0
        push!(grassy_cells, cell)
    end
end

grassy_cells
sample(grassy_cells)


if length(grassy_cells) > 0
   # sample 1 cell
   m_to = sample(grassy_cells)
   # move
   move_agent!(fish, m_to, model)
end

m_to = sample(collect(near_cells))

space = GridSpace((100, 100, 50), periodic = false)
waterfinder = AStar(space; diagonal_movement = true)

random_walkable(model_initilised, waterfinder)



# load csv of lake bathymetry ------------------------------------------------------
using CSV
using DataFrames


df = CSV.read("data\\taupo_500m.csv", DataFrame) |> Tables.matrix

# convert to Int
floor.(Int, convert.(Float64, df))



lake_url = "data\\taupo_500m.csv"

# lake topology
lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix
heightmap = floor.(Int, convert.(Float64, lake_mtx))

heightmap2 = heightmap .+ abs(minimum(heightmap)) .+ 1
heightmap2
dims = (size(heightmap2)..., 50)

heightmap2
heightmap2[heightmap2.=10155] .= 0

replace(heightmap2, 10155 => 0)



rng = MersenneTwister(1234)
# space = GridSpace(dims, periodic = false)

# Note that the dimensions of the space do not have to correspond to the dimensions of the heightmap ... dont understand how this works... 
# might only be for continuous spaces
#space = GridSpace((100, 100, 50), periodic = false)
space = GridSpace(dims, periodic = false)


swim_walkmap = BitArray(falses(dims...))

 # air animals can fly at any height upto lake_surface_level

for i in 1:dims[1], j in 1:dims[2]
    if heightmap2[i, j] < lake_surface_level
        swim_walkmap[i, j, (heightmap2[i, j]+1):lake_surface_level] .= true
    end
end



# define where "fish" can walk - everywhere
# land_walkmap = BitArray(trues(dims...))

=#