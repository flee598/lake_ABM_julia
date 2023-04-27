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

# CURRENTLY WORKING ON:
# SMELT MOVEMENT:
#       - SMELT RESOURCE CONSUMPTION
# EXPAND TO kOARO/trout


# load packages ---------------------------------------------------------------------
using Agents, Agents.Pathfinding
using Random
using StatsBase
using CSV                   # importing bathymetry from csv
using DataFrames            # importing bathymetry from csv
using InteractiveDynamics   # interactive plots
using GLMakie               # interactive plots
using FileIO: load        # used to download example heightmap

# define agents
@agent Fish GridAgent{3} begin
    type::Symbol
    length::Float64
end


# agent types
Trout(id, pos, length) = Fish(id, pos, :trout, length)
Koaro(id, pos, length) = Fish(id, pos, :koaro, length)
Smelt(id, pos, length) = Fish(id, pos, :smelt, length)

# Helper functions ----------------------------------------------------

# logistic growth - used for resources
function log_grow(r, K, pop)
    r * pop * (1 - (pop/K))
end

# initialize model ------------------------------------------------------------------------
function initialize_model(;
    #heightmap_url =
    #"https://raw.githubusercontent.com/JuliaDynamics/" *
    #"JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png", 
    lake_url = "data\\taupo_500m.csv",
    n_trout = 5,
    n_koaro = 10,
    n_smelt = 10,
    max_littoral_depth = 50,    # how far down does the littoral go in meters
    seed = 12345,)


    # example topology ----------------------------------------------------
    #heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1
    #heightmap2 = heightmap 
    # -----------------------------------------------------------------------

    # OR

    # load lake topology ----------------------------------------------------------
    lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix

    # convert to integer
    heightmap = floor.(Int, convert.(Float64, lake_mtx))


    # rescale so that there aren't negative depth values (the deepest point in the lake = 1)
    heightmap2 = heightmap .+ abs(minimum(heightmap)) .+ 1
    heightmap2 .= heightmap .* -1
    # -------------------------------------------------
    #


    # lake depth 
    mx_dpth = maximum(heightmap2)

    # create new lake_type variable -----------------------------------------------
    # 1 = littoral, 0 = pelagic
    lake_type = ones(Int, size(heightmap2))
    lake_type .= heightmap2

    # define the maximum depth of the littoral zone - this could be automated depending on env conditions

    # if take_type (depth) is between 0 and max_littoral_depth cell is littoral
    lake_type[lake_type .> -1 .&& lake_type .< (max_littoral_depth + 1)] .= 1

    # if lake is deeper than max_littoral_depth cell is pelagic
    lake_type[lake_type .> max_littoral_depth] .= 0


    # set limits between which agents can exist (surface and lake bed) ----------------
    # currently agents can get out of the water a bit!
    lake_surface_level = mx_dpth
    lake_floor = 2


    # lake dimensions ---------------------------------------------------------
    dims = (size(heightmap2)..., mx_dpth)

    # 3d version of lake type, used for indexing
    lake_type_3d = repeat(lake_type, 1,1,dims[3])

    # Note that the dimensions of the space do not have to correspond to the dimensions of the heightmap ... dont understand how this works... 
    # might only be for continuous spaces
    #space = GridSpace((100, 100, 50), periodic = false)
    space = GridSpace(dims, periodic = false)

    #  swimable space
    swim_walkmap = BitArray(falses(dims...))

    # fish can swim at any depth between the lake bed and lake suface


    for i in 1:dims[1], j in 1:dims[2]
        if lake_floor < heightmap2[i, j] < lake_surface_level
            swim_walkmap[i, j, (heightmap2[i, j]+1):lake_surface_level] .= true
        end
    end

    #=
    # for video - trout/smelt only in deep water
    for i in 1:dims[1], j in 1:dims[2]
        if lake_floor < heightmap2[i, j] < lake_surface_level
            swim_walkmap[i, j, (heightmap2[i, j]+1):lake_surface_level] .= true
        end
    end
    =#

    # moving on lake flooor - rather than anywhere - not used
    land_walkmap = BitArray(falses(dims...))
        for i in 1:dims[1], j in 1:dims[2]
            # land animals can only walk on top of the terrain between water_level and grass_level
            if lake_floor <= heightmap[i, j] < lake_surface_level
                land_walkmap[i, j, heightmap[i, j]+1] = true
            end
        end


    # create lake basal resource array
    lake_basal_resource = zeros(dims)

    # populate with basal resource
    # this is a 3d array, where each matrix slice is x/y and the z dim is depth 
    for i in 1:dims[1], j in 1:dims[2]
        if lake_type[i,j] == 0    # pelagic basal resource amount
            lake_basal_resource[i, j, 1:mx_dpth] .= 5.0
        end
        if lake_type[i,j] == 1 # littoral basal resource amount
            lake_basal_resource[i, j, 1:mx_dpth] .= 10.0 
        end
    end


    # model properties - see type stability issue in Agents.jl -> performance tips, might need to change this
    properties = Dict(
        :swim_walkmap => swim_walkmap,
        :landfinder => AStar(space; walkmap = land_walkmap),
        :waterfinder => AStar(space; walkmap = swim_walkmap, diagonal_movement = true),
        :heightmap2 => heightmap2,
        :lake_type => lake_type,
        :lake_type_3d => lake_type_3d,
        :lake_basal_resource => lake_basal_resource,
        :tick => 1::Int64,
    )



    # rng
    rng = MersenneTwister(seed)

    model = ABM(Fish, space; properties, rng, scheduler = Schedulers.randomly, warn = false)

    # Add agents -----------------------

    # trout
    for _ in 1:n_trout
        add_agent_pos!(
            Trout(
                nextid(model), 
                random_walkable(model, model.waterfinder),
                10.0,     # fish length
            ),
            model,
        )
    end

    # koaro
    for _ in 1:n_koaro
        add_agent_pos!(
            Koaro(
                nextid(model), 
                random_walkable(model, model.waterfinder),
                10.0,     # fish length
            ),
            model,
        )
    end

    # smelt
    for _ in 1:n_smelt
        add_agent_pos!(
            Smelt(
                nextid(model), 
                random_walkable(model, model.waterfinder),
                10.0,     # fish length
            ),
            model,
        )
    end

    return model

end

# fish movement --------------------------------------------------------------
function fish_step!(fish, model)
    if fish.type == :trout
        trout_step!(fish, model)
    elseif fish.type == :koaro
        koaro_step!(fish, model)
    else
        smelt_step!(fish, model)
    end
end

# koaro movement ------------------------------------------
function koaro_step!(koaro, model)
 
    # only move if there are no resources in current cell   ------- CURRENTLY CAUSES SMELT NOT TO MOVE - need to implement resource consumption
    if model.lake_basal_resource[koaro.pos...] < 10.0
            
        # 1 = vison range
        near_cells = nearby_positions(koaro.pos, model, 1)
        
        # find which of the nearby cells are allowed to be moved onto + have resources
        # ADD TROUT AVOIDANCE
        swimable_cells = []
        for cell in near_cells
            if model.swim_walkmap[cell...] > 0 && model.lake_basal_resource[cell...] > 2.0
                push!(swimable_cells, cell)
            end
        end
    
        # if there is a neighbouring cell with resources > 0, randomly select one to move to
        if length(swimable_cells) > 0
            
            # First, determine if near cells are pel or lit and generate a vector of weights
            # get resource type for grassy cells
            ind1  = [model.lake_type_3d[t...] for t in swimable_cells]
            ind1 = convert(Array{Float64, 1}, ind1)

            # pel and lit weights - 1 = pelagic, 0 = littoral
            resource_pref_adult = 0.5

            pel_p = resource_pref_adult
            lit_p = 1.0 - pel_p

            # convert pel/lit indicies to weights
            ind1[ind1 .== 0.0] .= pel_p
            ind1[ind1 .== 1.0] .= lit_p

            # randomly choose one of the nearby cells with resources, weighted by habitat preference
            m_to = sample(swimable_cells, Weights(ind1))
            # move
            move_agent!(koaro, m_to, model)

        else
            # if none of the near cells have resources, just pick one at random - NO LITTORAL/PELAGIC PREFERENCE
            swimable_cells2 = []
            for cell in near_cells
                if model.swim_walkmap[cell...] > 0
                    push!(swimable_cells2, cell)
                end
            end
            m_to = sample(collect(swimable_cells2))
            move_agent!(koaro, m_to, model)
        end
    end

    # NOTE YET IMPLEMNTED
    # eat!(koaro, model)


end

# smelt movement ------------------------------------------
# 1: if current cell has resources do nothing
# 2: otherwise, check which neighbouring cells have resources and are in moving range
# 3: weight possible moving cells by prefernce term and randmoly selct one

function smelt_step!(smelt, model)
     
    # only move if there are no resources in current cell   ------- CURRENTLY CAUSES SMELT NOT TO MOVE - need to implement resource consumption
    if model.lake_basal_resource[smelt.pos...] < 10.0
        
        # 1 = vison range
        near_cells = nearby_positions(smelt.pos, model, 1)
         
        # find which of the nearby cells are allowed to be moved onto + have resources
        # ADD TROUT AVOIDANCE
        swimable_cells = []
        for cell in near_cells
            if model.swim_walkmap[cell...] > 0 && model.lake_basal_resource[cell...] > 2.0
                push!(swimable_cells, cell)
            end
        end
     
        # if there is a neighbouring cell with resources > 0, randomly select one to move to
        if length(swimable_cells) > 0
            
            # First, determine if near cells are pel or lit and generate a vector of weights
            # get resource type for grassy cells
            ind1  = [model.lake_type_3d[t...] for t in swimable_cells]
            ind1 = convert(Array{Float64, 1}, ind1)

            # pel and lit weights - 1 = pelagic, 0 = littoral
            resource_pref_adult = 0.5

            pel_p = resource_pref_adult
            lit_p = 1.0 - pel_p

            # convert pel/lit indicies to weights
            ind1[ind1 .== 0.0] .= pel_p
            ind1[ind1 .== 1.0] .= lit_p

            # randomly choose one of the nearby cells with resources, weighted by habitat preference
            m_to = sample(swimable_cells, Weights(ind1))
            # move
            move_agent!(smelt, m_to, model)
        else
            # if none of the near cells have resources, just pick one at random - NO LITTORAL/PELAGIC PREFERENCE
            swimable_cells2 = []
            for cell in near_cells
                if model.swim_walkmap[cell...] > 0
                    push!(swimable_cells2, cell)
                end
            end

            m_to = sample(collect(swimable_cells2))
            move_agent!(smelt, m_to, model)
        end
    end

    # NOT YET IMPLEMNTED
    eat!(smelt, model)


end


# trout movement ------------------------------------------
function trout_step!(trout, model)
    
    # 1 = vison range
    near_cells = nearby_positions(trout.pos, model, 1)
    
    # storage
    swimable_cells = []
    
    # find which of the nearby cells are allowed to be moved onto
    for cell in near_cells
        if model.swim_walkmap[cell...] > 0
            push!(swimable_cells, cell)
        end
    end
    
    if length(swimable_cells) > 0
        # sample 1 cell
        m_to = sample(swimable_cells)
        # move
        move_agent!(trout, m_to, model)
    end
end



# define agent eating -----------------------------------------------------------------------------
function eat!(smelt, model)

    consume_amount = 50.0

    if model.lake_basal_resource[smelt.pos...] > 0.1     # if there are rsources available 
        #smelt.energy += smelt.Δenergy                    # give smelt energy
        #smelt.length += smelt.Δenergy                    # grow smelt

        # reduce resource amount - reduces resource by "consume_amount", unless less than that amount exists, then consume all
        fut_res = model.lake_basal_resource[smelt.pos...] - consume_amount

        if fut_res < 0.0 
            model.lake_basal_resource[smelt.pos...] -= abs(fut_res)
        else
            model.lake_basal_resource[smelt.pos...] -= consume_amount
        end

        # currently resources can go negative ... fix
    end
    return
end


# resource step - something mundane - TO BE UPDATED -------------------------------------
function lake_resource_step!(model)

    # will need to eventually add logistic growth
    # pelagic_growable[i] = round(pelagic_growable[i] + log_grow(0.2, 100, pelagic_growable[i]), digits = 3)  + 0.5 # add small constant, or else if it goes to 0 it will never regrow 

    model.lake_basal_resource[model.lake_type_3d .== 0] .= model.lake_basal_resource[model.lake_type_3d .== 0] .+ 10.00

    model.lake_basal_resource[model.lake_type_3d .== 1] .= model.lake_basal_resource[model.lake_type_3d .== 1] .+ 5.00

    model.tick += 1
end

# set up model -----------------------------------------------------------------------------
model_initilised = initialize_model() 


# plotting params -------------------------------------------------------------------------

# agnt colours
animalcolor(a) =
if a.type == :trout
        :yellow
elseif a.type == :koaro
        :red
else
        :blue
end


plotkwargs = (;
    ac = animalcolor,
    as = 2,
    am = :circle,
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
    frames = 150,
    framerate = 5,
    ac = animalcolor,
    as = 1,
    static_preplot!,
    title = "Lake food web. Trout: yellow, Smelt: blue, Koaro: red "
)


# record model output
model_initilised = initialize_model()
steps = 10
adata = [:pos, :type, :length]
mdata = [:tick, :lake_basal_resource]

# obtainer = copy - use this if you need to update the mdf output - by default if the output is mutable container it 
# won't show updates. using obtainer = copy will reduce performance, only use for prototyping 
adf, mdf = run!(model_initilised, fish_step!, lake_resource_step!, steps; adata, mdata, obtainer = deepcopy)
mdf
adf

mdf[:,3][2][:,:,1]
adf[:,3]

df1 = filter(:id => n -> n == 10, adf)
df1

mdf[:,3][1][17,4,101]
mdf[:,3][2][17,4,101]

mdf[:,3][4][16,5,102]
mdf[:,3][5][16,5,102]
mdf[:,3][6][16,5,102]
mdf[:,3][7][16,5,102]
mdf[:,3][8][16,5,102]
mdf[:,3][9][16,5,102]

mdf[:,3][10][16,6,103]
mdf[:,3][11][16,6,103]

mdf[:,3][1]

mdf[:,3][10]




# interactive plot with figures ------------------------------------

trout(a) = a.type == :trout
smelt(a) = a.type == :smelt
koaro(a) = a.type == :koaro


adata = [(trout, count), (smelt, count), (koaro, count)]

plotkwargs = (;
    ac = animalcolor,
    as = 2,
    am = :circle)

fig, p = abmexploration(model_initilised;
agent_step! = fish_step!,
model_step! = lake_resource_step!,
plotkwargs...,
adata,
alabels = ["Trout abund", "Smelt abund", "koaro abund"]
)

fig





# -------------------------- Testing -----------------------------------------------------------------------

lake_url = "data\\taupo_500m.csv"
lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix


# convert to integer
heightmap = floor.(Int, convert.(Float64, lake_mtx))


# rescale so that there aren't negative depth values (the deepest point in the lake = 1)
heightmap2 = heightmap .+ abs(minimum(heightmap)) .+ 1
heightmap2 .= heightmap .* -1

heightmap
heightmap2
# try removing lake walls so we can see what is going on in the videos - doesn't work as the littoral zone is much shallower
# than the pelagic so it obscures the view
#heightmap2 = replace(heightmap2, -9999 => 0)

# lake depth 
mx_dpth = maximum(heightmap2)

# create new lake_type variable -----------------------------------------------
# 1 = littoral, 0 = pelagic
lake_type = ones(Int, size(heightmap2))
lake_type .= heightmap2
lake_type

dims = (size(heightmap2)..., mx_dpth)

max_littoral_depth = 50

# if take_type (depth) is between 0 and max_littoral_depth cell is littoral
lake_type[lake_type .> -1 .&& lake_type .< (max_littoral_depth + 1)] .= 1

# if lake is deeper than max_littoral_depth cell is pelagic
lake_type[lake_type .> max_littoral_depth] .= 0

lake_type
# set limits between which agents can exist (surface and lake bed) ----------------
# currently agents can get out of the water a bit!
lake_surface_level = mx_dpth
lake_floor = 1


# ------------- testing set up -----------------------------------------------------


heightmap_url =
"https://raw.githubusercontent.com/JuliaDynamics/" *
"JuliaDynamics/master/videos/agents/rabbit_fox_hawk_heightmap.png"

max_littoral_depth = 50   # how far down does the littoral go in meters



# example topology
heightmap = floor.(Int, convert.(Float64, load(download(heightmap_url))) * 39) .+ 1

#lake_url = "data\\taupo_500m.csv"

# load lake topology ----------------------------------------------------------
# lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix

# convert to integer
# heightmap = floor.(Int, convert.(Float64, lake_mtx))


# rescale so that there aren't negative depth values (the deepest point in the lake = 1)
#heightmap2 = heightmap .+ abs(minimum(heightmap)) .+ 1
#heightmap2 .= heightmap .* -1


# lake depth 
mx_dpth = maximum(heightmap2)

# create new lake_type variable -----------------------------------------------
# 1 = littoral, 0 = pelagic
lake_type = ones(Int, size(heightmap2))
lake_type .= heightmap2

# define the maximum depth of the littoral zone - this could be automated depending on env conditions
