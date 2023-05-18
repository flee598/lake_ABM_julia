#=
3D model of a lake food web.

Overview:
1: generate a 3D lake environemnt from bathymetry data (currently csv)
2: populate lake with fish (3 species, trout (top predator), smelt and koaro (compeititors that consume resources produced)
   in cells) - fish have a variety of traits (e.g. length)
3: define lake cell traits (e.g. resource type (pelagic vs littoral), growth rate)
4: define fish movement (random in 3D, but weighted to move towards resources)
5: define fish eat
6: define fish reproduce 
7: define fish die
9: run for n model steps
=#

#= 
To do
If length < 5mm be pelagic (koaro and smelt), if larger, smelt pelagic, koaro littoral
large smelt preference prey term - small koaro then pelagic resource, then littoral resource

Trout predation
1: if prey in nearby cell move to cell (have preference term for smelt)
2: consume n prey (scale with length?)

koaro/smelt predator avoidence - can do last

=#


#using Agents
using Agents, Agents.Pathfinding
using Random
using Distributions   
using InteractiveDynamics
using GLMakie                   # for interactive plots
using StatsBase
using CSV                      # importing CSV
using DataFrames               # importing bathymetry from csv


# define agents -----------------------------------------------------------------------------
# geneic traits of all fish
@agent Fish GridAgent{3} begin
    type::Symbol                      # koaro, smelt, trout
    energy::Float64                   # current energy level
    length::Float64                   # length in mm
    stage::Int64                      # 0 = juvenile, 1 = adult
end

# Agent types
Smelt(id, pos, energy, length, stage) = Fish(id, pos, :smelt, energy, length, stage)
Koaro(id, pos, energy, length, stage) = Fish(id, pos, :koaro, energy, length, stage)
Trout(id, pos, energy, length, stage) = Fish(id, pos, :trout, energy, length, stage)


# Model properties - somewhere to store params - using mutable struct as mixed types, params defined below
Base.@kwdef mutable struct Parameters
    swim_walkmap::BitArray
    waterfinder::AStar
    heightmap2::Matrix{Int64}
	basal_resource::Array{Float64,3}
	basal_resource_type::Array{Int64,3}
	res_grow_r_lit::Float64
    res_grow_r_pel::Float64
    res_k_lit::Float64
    res_k_pel::Float64
    Δenergy_smelt::Float64
    consume_amount_smelt::Float64
    breed_prob_smelt::Float64     
    breed_mortality_smelt::Float64 
    growth_rate_smelt::Float64     
    length_mean_smelt::Float64
    length_sd_smelt::Float64
    vision_smelt::Int64
    n_juv_mean_smelt::Int64
    n_juv_sd_smelt::Int64
    size_maturity_smelt::Float64
    mortality_random_smelt::Float64
    resource_pref_adult_smelt::Float64
    resource_pref_juv_smelt::Float64
    size_mature_smelt::Float64
    fecundity_mean_smelt::Int64      
    fecundity_sd_smelt::Int64
    tick::Int64
end


# initialize model --------------------------------------------------------------------------

function initialize_model(;
    # dims = (15, 15, 15),              # grid size
    lake_url = "data\\taupo_500m.csv",  # lake data
    max_littoral_depth = 50,            # how far down does the littoral go in meters
    res_grow_r_lit = 0.01,              # cell resource growth rate - littoral resources - draw from distribution?
    res_grow_r_pel = 0.01,              # cell resource growth rate - pelagic resources - draw from distribution?
    res_k_lit = 100.0,                  # cell resource carrying capacity - littoral - draw from distribution?
    res_k_pel = 50.0,                   # cell resource carrying capacity - littoral - draw from distribution?
    n_smelt = 5,                        # initial number of smelt
    n_koaro = 5,                        # initial number of smelt
    n_trout = 5,                        # initial number of smelt
    Δenergy_smelt = 5.0,                # energy gained from eating resource - draw from distribution?
    consume_amount_smelt = 10.0,        # max amount consumed in 1 timestep - draw from distribution?
    breed_prob_smelt = 0.0,             # probability of spawning (during seasonal window only) - draw from distribution?
    breed_mortality_smelt = 0.5,        # probability of dying after spawning - draw from distribution?
    growth_rate_smelt = 1.0,            # mm growth / time step - draw from distribution?
    length_mean_smelt = 15.0,           # mean adult smelt length - used for setting initial lengths
    length_sd_smelt = 1.0,              # SD adult smelt length - used for setting initial lengths
    vision_smelt = 1,                   # number of cells smelt can "see"
    n_juv_mean_smelt = 100,             # mean number of juveniles produced by 1 female (note this is juveniles, not eggs)
    n_juv_sd_smelt = 10,                # SD number of juveniles produced by 1 female (note this is juveniles, not eggs)
    size_maturity_smelt = 10.0,         # size (mm) when smelt transition from juvenile to adult
    mortality_random_smelt = 0.0001,    # probability of random mortality each timestep - draw from distribution?
    resource_pref_adult_smelt = 0.99,   # adult smelt preference for (1) pelagic (0) littoral resources - but if koaro larvae present consume them
    resource_pref_juv_smelt = 0.99,     # juvenile preference for pelagic (1) or littoral (0) resources
    size_mature_smelt = 20.0,           # length (mm) when smelt can reproduce
    fecundity_mean_smelt = 100,         # mean number larvae produced
    fecundity_sd_smelt  = 10,           # SD number larvae produced
    seed = 23182,                       # rng seed
)

# define rng 
rng = MersenneTwister(seed)

 # load lake topology ----------------------------------------------------------
 lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix

 # convert to integer
 heightmap = floor.(Int, convert.(Float64, lake_mtx))

 # rescale so that there aren't negative depth values (the deepest point in the lake = 1)
 heightmap2 = heightmap .+ abs(minimum(heightmap)) .+ 1
 heightmap2 .= heightmap .* -1
 # -------------------------------------------------

 # lake depth 
 mx_dpth = maximum(heightmap2)

 # create new lake_type variable -----------------------------------------------
 # 1 = littoral, 0 = pelagic
 lake_type = ones(Int, size(heightmap2))
 lake_type .= heightmap2

 # if take_type (depth) is between 0 and max_littoral_depth cell is littoral
 lake_type[lake_type .> -1 .&& lake_type .< (max_littoral_depth + 1)] .= 1

 # if lake is deeper than max_littoral_depth cell is pelagic
 lake_type[lake_type .> max_littoral_depth] .= 0

 # set limits between which agents can exist (surface and lake bed) ----------------
 lake_surface_level = mx_dpth
 lake_floor = 0
 
 # lake dimensions ---------------------------------------------------------
 dims = (size(heightmap2)..., mx_dpth)

 # 3d version of lake type, used for indexing
 basal_resource_type = repeat(lake_type, 1,1, dims[3])

 # Note that the dimensions of the space do not have to correspond to the dimensions of the heightmap ... dont understand how this works... 
 # might only be for continuous spaces
 space = GridSpace(dims, periodic = false)

 #  swimable space ----------------------------------------------------
 swim_walkmap = BitArray(falses(dims...))

 # fish can swim at any depth between the lake bed and lake suface
 for i in 1:dims[1], j in 1:dims[2]
    if lake_floor < heightmap2[i, j] < lake_surface_level
        swim_walkmap[i, j, (lake_surface_level -  heightmap2[i,j]):lake_surface_level] .= true
    end
end

# create lake resource array
basal_resource = zeros(dims)

# populate with basal resource
# this is a 3d array, where each matrix slice is x/y and the z dim is depth 
for i in 1:dims[1], j in 1:dims[2]
   if lake_type[i,j] == 0    # initial pelagic resource amount - 
       basal_resource[i, j, 1:mx_dpth] .= (res_k_pel / 5.0) 
   end
   if lake_type[i,j] == 1 # initial littoral  resource amount
       basal_resource[i, j, 1:mx_dpth] .= (res_k_lit / 5.0) 
   end
end

# define the properties
properties = Parameters(
    swim_walkmap,
    AStar(space; walkmap = swim_walkmap, diagonal_movement = true),
    heightmap2,
    basal_resource,
    basal_resource_type,
    res_grow_r_lit,
    res_grow_r_pel,
    res_k_lit,
    res_k_pel,
    Δenergy_smelt,
    consume_amount_smelt,
    breed_prob_smelt,     
    breed_mortality_smelt,
    growth_rate_smelt,
    length_mean_smelt,
    length_sd_smelt, 
    vision_smelt,
    n_juv_mean_smelt,
    n_juv_sd_smelt,
    size_maturity_smelt, 
    mortality_random_smelt,
    resource_pref_adult_smelt,
    resource_pref_juv_smelt,
    size_mature_smelt,
    fecundity_mean_smelt,
    fecundity_sd_smelt,
    1)

# define model
model = ABM(Fish, space; properties, rng, scheduler = Schedulers.randomly, warn = false)

# define a normal distribution - used to determine inital fish length for smelt
sm_sz = Normal(model.length_mean_smelt, model.length_sd_smelt)

# Add agents ---------------------
# Add smelt
for _ in 1:n_smelt
               add_agent_pos!(
                Smelt(
                    nextid(model),                             # Using `nextid` prevents us from having to manually keep track # of animal IDs
                    random_walkable(model, model.waterfinder),
                    rand(model.rng, 1:100) - 1,                 # Fish starting energy level - somewhere between 1 and 99
                    round(rand(sm_sz), digits = 3),             # initial length
                    1),                                         # initial stage 
                model,
            )
end

# Add koaro
for _ in 1:n_koaro
    add_agent_pos!(
     Koaro(
         nextid(model),                             # Using `nextid` prevents us from having to manually keep track # of animal IDs
         random_walkable(model, model.waterfinder),
         rand(model.rng, 1:100) - 1,                 # Fish starting energy level - somewhere between 1 and 99
         round(rand(sm_sz), digits = 3),             # initial length
         1),                                         # initial stage 
     model,
 )
end

# Add trout
for _ in 1:n_trout
    add_agent_pos!(
     Trout(
         nextid(model),                             # Using `nextid` prevents us from having to manually keep track # of animal IDs
         random_walkable(model, model.waterfinder),
         rand(model.rng, 1:100) - 1,                 # Fish starting energy level - somewhere between 1 and 99
         round(rand(sm_sz), digits = 3),             # initial length
         1),                                         # initial stage 
     model,
 )
end

return model
end

sheepwolfgrass = initialize_model() 


# Fish movement - set up seperately for each species ---------------------------------------------------
function fish_step!(fish, model)
    if fish.type == :smelt
        smelt_step!(fish, model)
   # elseif fish.type == :koaro
    #    koaro_step!(fish, model)
    #else
     #   trout_step!(fish, model)
    end
end





# define agent movement --------------------------------------------------------------------------

# Smelt movement ---------
function smelt_step!(smelt::Fish, model)

    # check if current pos has resources, - will need to update to check if predator is present 
    # if current cell has resources (and no predator) don't do anything 

    # check resources in current pos, if none, move
     if model.basal_resource[smelt.pos...] < 1.0

        # get id of near cells - vision_range = range agent can "see"
        near_cells = nearby_positions(smelt.pos, model, model.vision_smelt)
        
        # storage
        swimable_cells = []
    
        # find which of the nearby cells are allowed to be moved into
        for cell in near_cells
            if model.swim_walkmap[cell...] > 0 && model.basal_resource[cell...] > 1.0
                push!(swimable_cells, cell)
            end
        end

        # if there is a neighbouring cell with resources > 0, randomly select one to move to
        if length(swimable_cells) > 0

            # First, determine if near cells are pel or lit and generate a vector of weights
            # get resource type for grassy cells
            ind1  = [model.basal_resource_type[t...] for t in swimable_cells]
            ind1 = convert(Array{Float64, 1}, ind1)

            # pel and lit weights
            pel_p = model.resource_pref_adult_smelt
            lit_p = 1.0 - pel_p

            # convert pel/lit indicies to weights
            ind1[ind1 .== 0.0] .= pel_p
            ind1[ind1 .== 1.0] .= lit_p

            # randomly choose one of the nearby cells with resources
            m_to = sample(swimable_cells, Weights(ind1))

            # move
            move_agent!(smelt, m_to, model)
        else
            # if none of the near cells have resources, just pick one at random - NO LITTORAL/PELAGIC PREFERENCE
            swimable_cells = []
    
            # find which of the nearby cells are allowed to be moved onto
            for cell in near_cells
                if model.swim_walkmap[cell...] > 0
                    push!(swimable_cells, cell)
                end
            end

            # NEED TO TEST THIS  -set all resources to 0 and see what happens
            move_agent!(smelt, sample(swimable_cells), model)
            #walk!(smelt, sample(near_cells.itr.iter, 1)[1], model)
        end
    end 

    # smelt loose energy after each step
    #smelt.energy -= 4.0

    #  if < 0.5 die
    #if smelt.energy < 0.5
    #    remove_agent!(smelt, model)
    #    return
    #end

    # smelt eating - see function below
    smelt_eat!(smelt, model)

    # reproduction every 5 ticks - just for testing
   
   #=
    if mod(model.tick, 5) == 0
        #if rand(model.rng) ≤ smelt.reproduction_prob
        reproduce_smelt!(model)
        #end
    end
    =#

 # adults die based on probability - high for testing - will this kill all agents or is it run run for each agent?
   # if rand(model.rng) < smelt.mortality_random
   #     kill_agent!(smelt, model)
   # end

   # if smelt.length > 50.0
   #     kill_agent!(smelt, model)
   # end
end

# define agent eating -----------------------------------------------------------------------------

# smelt eat
function smelt_eat!(smelt::Fish, model)

    if model.basal_resource[smelt.pos...] > 1.0     # if there are resources available 
        
        # reduce resource amount - reduces resource by "consume_amount", unless less than that amount exists, then consume all
        fut_res = model.basal_resource[smelt.pos...] -= model.consume_amount_smelt

        if fut_res < 0.0 
            model.basal_resource[smelt.pos...] -= abs(fut_res)
            smelt.energy += abs(fut_res)                   # give smelt energy
            smelt.length += abs(fut_res)                   # grow smelt
        else
            model.basal_resource[smelt.pos...] -= model.consume_amount_smelt
            smelt.energy += model.consume_amount_smelt                 # give smelt energy
            smelt.length += model.consume_amount_smelt                # grow smelt
            
        end
    end
    return
end

# koaro eat
function koaro_eat!(koaro::Fish, model)

    if model.basal_resource[koaro.pos...] > 1.0     # if there are rsources available 
        
        # reduce resource amount - reduces resource by "consume_amount", unless less than that amount exists, then consume all
        fut_res = model.basal_resource[koaro.pos...] - model.consume_amount_koaro

        if fut_res < 0.0 
            model.basal_resource[koaro.pos...] -= abs(fut_res)
            koaro.energy += abs(fut_res)                   # give koaro energy
            koaro.length += abs(fut_res)                   # grow koaro
        else
            model.basal_resource[koaro.pos...] -= model.consume_amount_koaro
            koaro.energy += model.consume_amount_koaro              # give koaro energy
            koaro.length += model.consume_amount_koaro             # grow koaro
            
        end
    end
    return
end



# trout eat --- TO DO ----------------
# --------------------------------------



# define agent reproduction --------------------------------------------------------------------------
# smelt reproduction ADD POST REPRODUCTION MORTLITY

function reproduce_smelt!(model)

    # how many juvs to add?
    fecund_smelt = Normal(model.n_juv_mean_smelt, model.n_juv_sd_smelt)
    n_juv = round(rand(fecund_smelt), digits = 1)
    
    for _ in 1:n_juv
        add_agent_pos!(
         Smelt(
             nextid(model),                               # Using `nextid` prevents us from having to manually keep track # of animal IDs
             random_walkable(model, model.waterfinder),
             rand(model.rng, 1:20) - 1,                   # Fish starting energy level - somewhere between 1 and 99
             2,                                           # initial length 2 mm
             0),                                          # initial stage juv
         model,
     )
    end
end



# define cell resource growth  --------------------------------------------------------------------------------
# follows discrete time logistic growth

function grass_step!(model)

    # pelagic
    growable_pel = view(
        model.basal_resource,
        model.basal_resource_type .== 0,
    )
    growable_pel .= growable_pel .+ (model.res_grow_r_pel .* growable_pel .* (1.0 .- (growable_pel ./ model.res_grow_r_pel)))

    # littoral
    growable_lit = view(
        model.basal_resource,
        model.basal_resource_type .== 1,
    )
    growable_lit .= growable_lit .+ (model.res_grow_r_lit .* growable_lit .* (1.0 .- (growable_lit ./ model.res_k_lit)))

    # model counter
    model.tick += 1
end



# define plotting vars -----------------------------------------------------------------------------
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

sheepwolfgrass = initialize_model(seed = 22) 

# interactive plot ------------------------------------------------------------------------

# forces GLMakie to open a seperate plot window, rather than using 
# the VSCode plotting window ... needed for interactive plots
GLMakie.activate!(inline=false)

fig, ax, abmobs = abmplot(sheepwolfgrass;
    agent_step! = fish_step!,
    model_step! = grass_step!,
plotkwargs...)

fig

# data collection -------------------------------------------------------------------------------------

sheepwolfgrass = initialize_model(res_grow_r_lit = 0.05, res_grow_r_pel = 0.05)
steps = 100
adata = [:pos, :energy, :length, :type]

mdata = [:basal_resource, :basal_resource_type, :tick]

# obtainer = copy - use this if you need to update the mdf output - by default if the output is mutable container it 
# won't show updates. using obtainer = copy will reduce performance, only use for prototyping 
adf, mdf = run!(sheepwolfgrass, fish_step!, grass_step!, steps; adata, mdata, obtainer = deepcopy)


mdf


mdf[1,2]


# plot basl resources at a single depth
pltm = mdf[5,2]
pltm1 = pltm[:,:,5]

using Plots

Plots.heatmap(1:size(pltm1,2),
        1:size(pltm1,1),
        pltm1,
    c=cgrad([:blue, :yellow]),
    xlabel="x values", ylabel="y values"
)




# basal resources


# plot basal resourse in a single cell through time

y = [mdf[i,2][5,10,1] for i in axes(mdf,1)]
x = 1:length(y)

Plots.plot(x, y)


