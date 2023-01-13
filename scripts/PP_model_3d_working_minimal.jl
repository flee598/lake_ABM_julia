using Agents, Agents.Pathfinding
#using Agents
using Random
using InteractiveDynamics
using GLMakie
using FileIO: load
using StatsBase

# load csv of lake bathymetry ------------------------------------------------------
using CSV
using DataFrames


@agent Sheep GridAgent{3} begin
end


# initialize model ------------------------------------------------------------------------
function initialize_model(;
    #heightmap_url =
    #"https://raw.githubusercontent.com/JuliaDynamics/" *
    #"JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png", 
    lake_url = "data\\taupo_500m.csv",
    n_sheep = 10,
    #mountain_level = 200,
   # dims = (20, 20, 20),
    seed = 23182,
)

# example topology
#heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1

# lake topology
lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix
heightmap = floor.(Int, convert.(Float64, lake_mtx))

heightmap2 = heightmap .+ abs(minimum(heightmap)) .+ 1

# try removing lake walls so we can see what is going on
#heightmap2 = replace(heightmap2, 10155 => 0)

mx_dpth = abs(minimum(heightmap2)) + 10
#mx_dpth = 200

# set limits between which agents can exist (surface and lake bed)
mountain_level = mx_dpth - 10
#lake_floor = 1

dims = (size(heightmap2)..., mx_dpth)


rng = MersenneTwister(seed)
# space = GridSpace(dims, periodic = false)

# Note that the dimensions of the space do not have to correspond to the dimensions of the heightmap ... dont understand how this works... 
# might only be for continuous spaces
#space = GridSpace((100, 100, 50), periodic = false)
space = GridSpace(dims, periodic = false)


air_walkmap = BitArray(falses(dims...))

 # air animals can fly at any height upto mountain_level

for i in 1:dims[1], j in 1:dims[2]
    if heightmap2[i, j] < mountain_level
    #if lake_floor < heightmap2[i, j] < mountain_level
        air_walkmap[i, j, (heightmap2[i, j]+1):mountain_level] .= true
    end
end

# model properties
properties = (
    air_walkmap = air_walkmap,
    airfinder = AStar(space; walkmap = air_walkmap, diagonal_movement = true),
    heightmap2 = heightmap2,
    fully_grown = falses(dims),
)

model = ABM(Sheep, space; properties, rng, scheduler = Schedulers.randomly, warn = false
)

# Add agents
for _ in 1:n_sheep
    add_agent_pos!(
        Sheep(
            nextid(model), 
            random_walkable(model, model.airfinder),
        ),
        model,
    )
end


 # params here must be in the same order as the @agent section above
 #for _ in 1:n_sheep
 #    add_agent!(Sheep, model)
 #end


# Add grass with random initial growth
for p in positions(model)
    fully_grown = rand(model.rng, Bool)
    model.fully_grown[p...] = fully_grown
end

return model

end

# sheep movement - random ----------------------------------------------------
function sheepwolf_step!(sheep::Sheep, model)
    
    #walk!(sheep, rand, model)

    # plan where to move based on allowable space (must stay in lake)
    # 1 = vision range

    #=
    plan_route!(
                sheep,
                nearby_walkable(sheep.pos, model, model.airfinder, 1),
                model.airfinder
    )

    move_along_route!(sheep, model, model.airfinder)
 
 =#
# 1 = vison range
 near_cells = nearby_positions(sheep.pos, model, 1)

# storage
 grassy_cells = []
 
 
 # find which of the nearby cells are allowed to be moved onto
 for cell in near_cells
     if model.air_walkmap[cell...] > 0
         push!(grassy_cells, cell)
     end
 end


 if length(grassy_cells) > 0
    # sample 1 cell
    m_to = sample(grassy_cells)
    # move
    move_agent!(sheep, m_to, model)
 end



end


# resource - something mundane -----------------------------------------------------------
function grass_step!(model)
    for p in positions(model)
                model.fully_grown[p...] = true
            end
end


# set up model - fails -------------------------------------------------------------------
sheepwolfgrass = initialize_model() 





# plotting params -------------------------------------------------------------------------
plotkwargs = (;
    ac = :red,
    as = 2,
    am = :circle,
    scatterkwargs = (strokewidth = 1.0, strokecolor = :black),
)

# interactive plot ------------------------------------------------------------------------
fig, ax, abmobs = abmplot(sheepwolfgrass;
    agent_step! = sheepwolf_step!,
    model_step! = grass_step!,
plotkwargs...)
fig



# rd video with bathymetry surface added ---------------------------------------------
function static_preplot!(ax, model)
    surface(
        ax,
        #(100/205):(100/205):100,
        #(100/205):(100/205):100,
        model.heightmap2;
        colormap = :viridis
    )
end



animalcolor(a) = :red

abmvideo(
    "rabbit_fox_hawk.mp4",
    sheepwolfgrass, sheepwolf_step!, grass_step!;
    figure = (resolution = (1920 , 1080),),
    frames = 40,
    ac = animalcolor,
    as = 1,
    static_preplot!,
    title = "Rabbit Fox Hawk with pathfinding"
)



















heightmap_url =
    "https://raw.githubusercontent.com/JuliaDynamics/" *
    "JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png" 

heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
dims = (size(heightmap)..., 50)

air_walkmap = BitArray(falses(dims...))

mountain_level = 35

for i in 1:dims[1], j in 1:dims[2]
    if heightmap[i, j] < mountain_level
        air_walkmap[i, j, (heightmap[i, j]+1):mountain_level] .= true
    end
end


heightmap
air_walkmap[:, :, 20]


near_cells = nearby_positions((1,1,2), sheepwolfgrass, 1)
near_cells


# storage
grassy_cells = []
 
# find which of the nearby cells are allowed to be moved onto
for cell in near_cells
    if sheepwolfgrass.air_walkmap[cell...] > 0
        push!(grassy_cells, cell)
    end
end

grassy_cells
sample(grassy_cells)


if length(grassy_cells) > 0
   # sample 1 cell
   m_to = sample(grassy_cells)
   # move
   move_agent!(sheep, m_to, model)
end

m_to = sample(collect(near_cells))

space = GridSpace((100, 100, 50), periodic = false)
airfinder = AStar(space; diagonal_movement = true)

random_walkable(sheepwolfgrass, airfinder)



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


air_walkmap = BitArray(falses(dims...))

 # air animals can fly at any height upto mountain_level

for i in 1:dims[1], j in 1:dims[2]
    if heightmap2[i, j] < mountain_level
        air_walkmap[i, j, (heightmap2[i, j]+1):mountain_level] .= true
    end
end



# define where "sheep" can walk - everywhere
# land_walkmap = BitArray(trues(dims...))