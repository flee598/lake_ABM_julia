#=
3D model of a lake food web.

Overview:
1: generate a 3D lake environemnt from bathymetry data (currently csv)
2: populate lake with fish (3 species, trout (top predator), smelt and koaro (compeititors that consume resources produced
   in cells) - fish have a variety of traits (e.g. length, energy)
3: define lake cell traits (e.g. resource type (pelagic vs littoral), resource growth rate)
4: define fish movement (random in 3D, but weighted to move towards specific resource)
5: define fish eat
6: define fish reproduce 
7: define fish die
8: run model!
=#

#= 
To do

Trout predation
1: if prey in nearby cell move to cell (have preference term for smelt)
2: consume n prey (scale with length?)


koaro/smelt predator avoidence - can do last
=#

# load packages ------------------------------------------------------------------------------
using Random
using Distributions
using StatsBase
using Agents, Agents.Pathfinding       # ABM
using InteractiveDynamics              # ABM
using GLMakie                          # for interactive plots
using Plots                            # standard plots
using CSV                              # importing bathymetry from csv
using DataFrames                       # importing bathymetry from csv

# define agents -----------------------------------------------------------------------------

#  traits of all fish
@agent Fish GridAgent{3} begin
    type::Symbol                      # koaro, smelt, trout
    energy::Float64                   # current energy level
    length::Float64                   # length in mm
    stage::Int64                      # 0 = juvenile, 1 = adult
end


# Define agent types 
Smelt(id, pos, energy, length, stage) = Fish(id, pos, :smelt, energy, length, stage)
Koaro(id, pos, energy, length, stage) = Fish(id, pos, :koaro, energy, length, stage)
Trout(id, pos, energy, length, stage) = Fish(id, pos, :trout, energy, length, stage)

# Model properties ----------------------------------------------------------------------------
# somewhere to store params - using mutable struct as mixed types, params defined below
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
    consume_amount_koaro::Float64
    breed_prob_smelt::Float64     
    breed_mortality_smelt::Float64 
    growth_rate_smelt::Float64     
    length_mean_smelt::Float64
    length_sd_smelt::Float64
    vision_smelt::Int64
    vision_koaro::Int64
    n_juv_mean_smelt::Int64
    n_juv_sd_smelt::Int64
    size_maturity_smelt::Float64
    mortality_random_smelt::Float64
    resource_pref_adult_smelt::Float64
    resource_pref_juv_smelt::Float64
    resource_pref_adult_koaro::Float64
    resource_pref_juv_koaro::Float64
    size_mature_smelt::Float64
    fecundity_mean_smelt::Int64      
    fecundity_sd_smelt::Int64
    tick::Int64
end


# initialize model --------------------------------------------------------------------------

function initialize_model(;
    # dims = (15, 15, 15),              # grid size - can use for testing
    lake_url = "data\\taupo_500m.csv",  # lake data
    max_littoral_depth = 100,           # how far down does the littoral go in meters
    res_grow_r_lit = 0.01,              # cell resource growth rate - littoral resources - draw from distribution?
    res_grow_r_pel = 0.01,              # cell resource growth rate - pelagic resources - draw from distribution?
    res_k_lit = 100.0,                  # cell resource carrying capacity - littoral - draw from distribution?
    res_k_pel = 50.0,                   # cell resource carrying capacity - littoral - draw from distribution?
    n_smelt = 5,                        # initial number of smelt
    n_koaro = 5,                        # initial number of smelt
    n_trout = 5,                        # initial number of smelt
    Δenergy_smelt = 5.0,                # energy gained from eating resource - draw from distribution?
    consume_amount_smelt = 10.0,        # max amount consumed in 1 timestep - draw from distribution?
    consume_amount_koaro = 10.0,         # max amount consumed in 1 timestep - draw from distribution?
    breed_prob_smelt = 0.0,             # probability of spawning (during seasonal window only) - draw from distribution?
    breed_mortality_smelt = 0.95,        # probability of dying after spawning - draw from distribution?
    growth_rate_smelt = 1.0,            # mm growth / time step - draw from distribution?
    length_mean_smelt = 15.0,           # mean adult smelt length - used for setting initial lengths
    length_sd_smelt = 1.0,              # SD adult smelt length - used for setting initial lengths
    vision_smelt = 1,                   # number of cells smelt can "see"
    vision_koaro = 1,                   # number of cells koaro can "see"
    n_juv_mean_smelt = 100,             # mean number of juveniles produced by 1 female (note this is juveniles, not eggs)
    n_juv_sd_smelt = 10,                # SD number of juveniles produced by 1 female (note this is juveniles, not eggs)
    size_maturity_smelt = 10.0,         # size (mm) when smelt transition from juvenile to adult
    mortality_random_smelt = 0.01/365,    # probability of random mortality each timestep
    resource_pref_adult_smelt = 0.99,   # adult smelt preference for (1) pelagic (0) littoral resources - but if koaro larvae present consume them
    resource_pref_juv_smelt = 0.99,     # juvenile preference for pelagic (1) or littoral (0) resources
    resource_pref_adult_koaro = 0.01,   # adult koaro preference for (1) pelagic (0) littoral resources 
    resource_pref_juv_koaro = 0.99,     # juv koaro preference for (1) pelagic (0) littoral resources 
    size_mature_smelt = 20.0,           # length (mm) when smelt can reproduce
    fecundity_mean_smelt = 3,         # mean number larvae produced
    fecundity_sd_smelt  = 0.2,           # SD number larvae produced
    seed = 23182,                       # rng seed
)

# define rng 
rng = MersenneTwister(seed)

 # load lake topology ---------------------------------------
 lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix

 # convert to integer
 heightmap = floor.(Int, convert.(Float64, lake_mtx))

 # rescale so that there aren't negative depth values (the deepest point in the lake = 1)
 heightmap2 = heightmap .+ abs(minimum(heightmap)) .+ 1
 heightmap2 .= heightmap .* -1

 # lake depth 
 mx_dpth = maximum(heightmap2)

 # create new lake_type variable ----------------------------
 # 1 = littoral, 0 = pelagic
 lake_type = ones(Int, size(heightmap2))
 lake_type .= heightmap2

 # if lake_type (depth) is between 0 and max_littoral_depth cell is littoral
 lake_type[lake_type .> -1 .&& lake_type .< (max_littoral_depth + 1)] .= 1

 # if lake is deeper than max_littoral_depth cell is pelagic
 lake_type[lake_type .> max_littoral_depth] .= 0

 # set limits between which agents can exist (surface and lake bed) ----------------
 lake_surface_level = mx_dpth
 lake_floor = 0
 
 # lake dimensions ---------------------------------------------------------
 dims = (size(heightmap2)..., mx_dpth)

 # Note that the dimensions of the space do not have to correspond to the
 # dimensions of the heightmap ... dont understand how this works... 
 # might only be for continuous spaces
 space = GridSpace(dims, periodic = false)

 # 3d version of lake type, used for indexing
 basal_resource_type = repeat(lake_type, 1,1, dims[3])

 #  swimable space ----------------------------------------------------
 swim_walkmap = BitArray(falses(dims...))

 # fish can swim at any depth between the lake bed and lake suface
 for i in 1:dims[1], j in 1:dims[2]
    if lake_floor < heightmap2[i, j] < lake_surface_level
        swim_walkmap[i, j, (lake_surface_level -  heightmap2[i,j]):lake_surface_level] .= true
    end
end

# create lake resource array --------------------------------------
basal_resource = zeros(dims)

# populate with basal resource - varies depending on if littoral or pelgic
for i in 1:dims[1], j in 1:dims[2]
   if lake_type[i,j] == 0    # initial pelagic resource amount - 
       basal_resource[i, j, 1:mx_dpth] .= (res_k_pel / 5.0) 
   end
   if lake_type[i,j] == 1 # initial littoral  resource amount
       basal_resource[i, j, 1:mx_dpth] .= (res_k_lit / 5.0) 
   end
end

# define the properties -----------------------------------------
properties = Parameters(
    swim_walkmap,
    AStar(space; walkmap = swim_walkmap, diagonal_movement = true),  # dont think I need this as I'm not doing Pathfinding
    heightmap2,
    basal_resource,
    basal_resource_type,
    res_grow_r_lit,
    res_grow_r_pel,
    res_k_lit,
    res_k_pel,
    Δenergy_smelt,
    consume_amount_smelt,
    consume_amount_koaro,
    breed_prob_smelt,     
    breed_mortality_smelt,
    growth_rate_smelt,
    length_mean_smelt,
    length_sd_smelt, 
    vision_smelt,
    vision_koaro,
    n_juv_mean_smelt,
    n_juv_sd_smelt,
    size_maturity_smelt, 
    mortality_random_smelt,
    resource_pref_adult_smelt,
    resource_pref_juv_smelt,
    resource_pref_adult_koaro,
    resource_pref_juv_koaro,
    size_mature_smelt,
    fecundity_mean_smelt,
    fecundity_sd_smelt,
    1)

# define model -----------------------------------
model = ABM(Fish, space; properties, rng, scheduler = Schedulers.randomly, warn = false)

# define a normal distribution - used to determine inital fish length for smelt
sm_sz = Normal(model.length_mean_smelt, model.length_sd_smelt)

# Add agents -------------------------------------
# Add smelt
for _ in 1:n_smelt
               add_agent_pos!(
                Smelt(
                    nextid(model),                              # Using `nextid` prevents us from having to manually keep track # of animal IDs
                    random_walkable(model, model.waterfinder),  # random initial position
                    rand(model.rng, 1:100) - 1,                 # Fish starting energy level
                    round(rand(sm_sz), digits = 3),             # initial length
                    1),                                         # initial stage 
                model,
            )
end

# Add koaro
for _ in 1:n_koaro
    add_agent_pos!(
     Koaro(
         nextid(model),
         random_walkable(model, model.waterfinder),  # random initial position
         rand(model.rng, 1:100) - 1,                 # Fish starting energy level
         #round(rand(sm_sz), digits = 3),             # initial length
         15.0,                                          # initial length - TESTING
         1),                                           # initial stage 
     model,
 )
end

# Add trout
for _ in 1:n_trout
    add_agent_pos!(
     Trout(
         nextid(model),                             
         random_walkable(model, model.waterfinder),  # random initial position
         rand(model.rng, 1:100) - 1,                 # Fish starting energy level
         round(rand(sm_sz), digits = 3),             # initial length
         1),                                         # initial stage 
     model,
 )
end

return model
end

# test initialise works
initialised_model = initialize_model() 


# Fish movement - set up seperately for each species ---------------------------------------------------
function fish_step!(fish, model)
    if fish.type == :smelt
        smelt_step!(fish, model)
  #  elseif fish.type == :koaro
   #     koaro_step!(fish, model)
    #else
     #   trout_step!(fish, model)
    end
end




# define agent movement --------------------------------------------------------------------------

# Smelt movement ----------------------
function smelt_step!(smelt::Fish, model)
    
    # find any juvenile koaro prey nearby
    koaro_prey = [
        x.pos for x in nearby_agents(smelt, model, model.vision_smelt) if
            x.type == :koaro && x.length < 10.0
            ]

     # if there are juvenile koaro prey nearby move to them  - only if the smelt is an adult   
    if !isempty(koaro_prey) && smelt.length > 10.0
    
        # need to add weight - move to cell with the most prey
        move_agent!(smelt, sample(koaro_prey), model)
   
     # if no prey koaro nearby, or smelt is juv look for cell resources
     # check if current pos has resources
     # if current cell has resources (and no predator) don't do anything if none, move
    elseif model.basal_resource[smelt.pos...] < 100.0                          # TESTING VALUE

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

        # if there is a neighbouring cell with resources > 0, randomly select one to move to,
        # weighted by resoruce preference
        if !isempty(swimable_cells)

            # First, determine if near cells are pel or lit and generate a vector of weights
            
            # get resource type for swimable_cells
            ind1  = [model.basal_resource_type[t...] for t in swimable_cells]
            ind1 = convert(Array{Float64, 1}, ind1)

             # pel and lit weights - depends on fish size
             if smelt.length < 10.0
                pel_res_pref = model.resource_pref_juv_smelt
            else
                pel_res_pref = model.resource_pref_adult_smelt
            end
            
            # littoral pref
            lit_p = 1.0 - pel_res_pref

            # convert pel/lit indicies to weights
            ind1[ind1 .== 0.0] .= pel_res_pref
            ind1[ind1 .== 1.0] .= lit_p

            # randomly choose one of the nearby cells with resources and move
            move_agent!(smelt, sample(swimable_cells, Weights(ind1)), model)
        else
            # if none of the near cells have resources, just pick one at random - NO LITTORAL/PELAGIC PREFERENCE
            swimable_cells = []
    
            # find which of the nearby cells are allowed to be moved onto
            for cell in near_cells
                if model.swim_walkmap[cell...] > 0
                    push!(swimable_cells, cell)
                end
            end

            # NEED TO TEST THIS - set all resources to 0 and see what happens
            move_agent!(smelt, sample(swimable_cells), model)
        end
    end 

    # smelt eating - see function below
    smelt_eat!(smelt, model)



    # reproduction every 5 ticks - just for testing
    # in reality it will need to be between days 300 - 360 for example

    if mod(model.tick, 5) == 0 &&  smelt.length > 40.0                       # TESTING VALUE
        #if rand(model.rng) ≤ smelt.reproduction_prob
        reproduce_smelt!(smelt, model)
        #end
    end


    # smelt loose energy after each step
    smelt.energy -= 0.001                                                    # TESTING VALUE

    # if energy < 0.1 die - if there are very little resources smelt starve
    if smelt.energy < 1.0
        remove_agent!(smelt, model)
        return
    end

    # adults die based on random probability
    if rand(model.rng) < model.mortality_random_smelt
        remove_agent!(smelt, model)
    end

   # either implemnt age or size based motality? might not need if we have random mortality...
   #if smelt.length > 50.0
   #    kill_agent!(smelt, model)
   #end


end




# koaro movement ----------------------
function koaro_step!(koaro::Fish, model)

    # check if current pos has resources, - will need to update to check if predator is present 
    # if current cell has resources (and no predator) don't do anything 

    # check resources in current pos, if none, move
     if model.basal_resource[koaro.pos...] < 10.0

        # get id of near cells - vision_range = range agent can "see"
        near_cells = nearby_positions(koaro.pos, model, model.vision_koaro)

        # storage
        swimable_cells = []
    
        # find which of the nearby cells are allowed to be moved into
        for cell in near_cells
            if model.swim_walkmap[cell...] > 0 && model.basal_resource[cell...] > 1.0
                push!(swimable_cells, cell)
            end
        end 

        # if there is a neighbouring cell with resources > 0, randomly select one to move to,
        # weighted by resoruce preference
        if !isempty(swimable_cells)

            # First, determine if near cells are pel or lit and generate a vector of weights
            
            # get resource type for swimable_cells
            ind1  = [model.basal_resource_type[t...] for t in swimable_cells]
            ind1 = convert(Array{Float64, 1}, ind1)

            # pel and lit weights - depends on fish size
            if koaro.length < 10.0
                pel_res_pref = model.resource_pref_juv_koaro
            else
                pel_res_pref = model.resource_pref_adult_koaro
            end

            # littoral weight
            lit_p = 1.0 - pel_res_pref

            # convert pel/lit indicies to weights
            ind1[ind1 .== 0.0] .= pel_res_pref
            ind1[ind1 .== 1.0] .= lit_p

            # randomly choose one of the nearby cells with resources
            m_to = sample(swimable_cells, Weights(ind1))

            # move
            move_agent!(koaro, m_to, model)
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
            move_agent!(koaro, sample(swimable_cells), model)

        end
    end 

    # koaro loose energy after each step
    #koaro.energy -= 4.0

    #  if < 0.5 die
    #if koaro.energy < 0.5
    #    remove_agent!(koaro, model)
    #    return
    #end

    # koaro eating - see function below
    koaro_eat!(koaro, model)

    # reproduction every 5 ticks - just for testing
   
   #=
    if mod(model.tick, 5) == 0
        #if rand(model.rng) ≤ koaro.reproduction_prob
        reproduce_smelt!(model)
        #end
    end
    =#

 # adults die based on probability - high for testing - will this kill all agents or is it run run for each agent?
   # if rand(model.rng) < koaro.mortality_random
   #     kill_agent!(koaro, model)
   # end

   # if koaro.length > 50.0
   #     kill_agent!(koaro, model)
   # end
end

5+5
# trout movement --- TO DO ----------------


# define agent eating -----------------------------------------------------------------------------

# smelt eat - koaro larvae, then cell resources
function smelt_eat!(smelt::Fish, model)

    # check if there is koaro fry in current cell, if so eat them
    food = [x for x in agents_in_position(smelt, model) if x.type == :koaro && x.length < 10.0]
    die_it = collect(food)
    
    # perference is to eat koaro fry, if not eat cell resource
    if !isempty(die_it)

        # determine how many koaro fry to eat (in one tick), increases with smelt size
        # if only a few fry presnt, eat all, otherwise scale with smelt length
        if length(die_it)  < round(smelt.length * 1)
            n_eat = length(die_it)
        else
            n_eat = round(smelt.length * 1)
        end

        # remove eaten koaro 
        remove_all!(model, StatsBase.sample(model.rng, die_it, n_eat, replace = false))

        # give smelt energy
        smelt.energy +=  (n_eat * model.Δenergy_smelt)
         

    # if there are resources available 
    elseif model.basal_resource[smelt.pos...] > 1.0     
        
        # reduce resource amount - reduces resource by "consume_amount", unless less than that amount exists, then consume all
        fut_res = model.basal_resource[smelt.pos...] -= model.consume_amount_smelt
        
        if fut_res < 0.0 
            model.basal_resource[smelt.pos...] -= abs(fut_res)
            smelt.energy += abs(fut_res)                   # give smelt energy
            smelt.length += abs(fut_res)                   # grow smelt
        else
            model.basal_resource[smelt.pos...] -= model.consume_amount_smelt
            smelt.energy += model.consume_amount_smelt                 # give smelt energy
            smelt.length += model.consume_amount_smelt                 # grow smelt  
        end
        
    end
   # return
end


# koaro eat - just cell resources
function koaro_eat!(koaro::Fish, model)

    if model.basal_resource[koaro.pos...] > 1.0     # if there are resources available 
        
        # reduce resource amount - reduces resource by "consume_amount", unless less than that amount exists, then consume all
        fut_res = model.basal_resource[koaro.pos...] -= model.consume_amount_koaro

        if fut_res < 0.0 
            model.basal_resource[koaro.pos...] -= abs(fut_res)
            koaro.energy += abs(fut_res)                   # give koaro energy
            #koaro.length += abs(fut_res)                   # grow koaro               # NO KOARAO GROWTH TESTING
        else
            model.basal_resource[koaro.pos...] -= model.consume_amount_koaro
            koaro.energy += model.consume_amount_koaro                 # give koaro energy
            #koaro.length += model.consume_amount_koaro                # grow koaro          # NO KOARAO GROWTH TESTING
            
        end
    end
    return
end


# trout eat --- TO DO ----------------



# define agent reproduction ----------------------------------------------------------------------------------- TO DO

# smelt reproduction
function reproduce_smelt!(smelt::Fish, model)

    # how many juvs to add?
    sm_spwn = Normal(model.fecundity_mean_smelt, model.fecundity_sd_smelt)
    n_juv = round(rand(sm_spwn), digits = 1)
    
    # add fry
    for _ in 1:n_juv
        add_agent_pos!(
         Smelt(
             nextid(model),                               # Using `nextid` prevents us from having to manually keep track # of animal IDs
             random_walkable(model, model.waterfinder),   # random starting location - could make pelagic
             rand(model.rng, 1:20) - 1,                   # Fish starting energy level - somewhere between 1 and 99
             2,                                           # initial length 2 mm
             0),                                          # initial stage juv
         model,
         )
    end

    # post spwaning mortality of adult
    if rand(model.rng) < model.breed_mortality_smelt
        remove_agent!(smelt, model)
    end

end



# koaro reproduction
function reproduce_kaoro!(model)

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

# trout reproduction
function reproduce_trout!(model)

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
    growable_pel .= growable_pel .+ (model.res_grow_r_pel .* growable_pel .* (1.0 .- (growable_pel ./ model.res_k_pel)))

    # littoral
    growable_lit = view(
        model.basal_resource,
        model.basal_resource_type .== 1,
    )
    growable_lit .= growable_lit .+ (model.res_grow_r_lit .* growable_lit .* (1.0 .- (growable_lit ./ model.res_k_lit)))

    # model counter
    model.tick += 1
end



# ---------------------------------------------------------------------------------------------------------------
# -------------------------------------- MODEL END --------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------



# TESTING / PLOTTING ETC

# interactive plot ------------------------------------------------------------------------

initialised_model = initialize_model() 

# agnt colours
animalcolor(a) =
if a.type == :trout
        :yellow
elseif a.type == :koaro
        :red
else
        :blue
end

# plot bits
plotkwargs = (;
    ac = animalcolor,
    as = 2,
    am = :circle,
)


# forces GLMakie to open a seperate plot window, rather than using 
# the VSCode plotting window ... needed for interactive plots
GLMakie.activate!(inline=false)

fig, ax, abmobs = abmplot(initialised_model;
    agent_step! = fish_step!,
    model_step! = grass_step!,
plotkwargs...)

fig

# data collection -------------------------------------------------------------------------------------

initialised_model = initialize_model()
steps = 10

# somewhere to store results
adata = [:pos, :energy, :length, :type]
mdata = [:basal_resource, :basal_resource_type, :tick]

# obtainer = copy - use this if you need to update the mdf output - by default if the output is mutable container it 
# won't show updates. using obtainer = copy will reduce performance, only use for prototyping 
adf, mdf = run!(initialised_model, fish_step!, grass_step!, steps; adata, mdata, obtainer = deepcopy)


# agent data
adf

adf[adf.type .== :smelt, :]


5*50*50

# cell data
mdf
mdf[2,2]




# plot basl resource at a random depth -----------------
plt_res = mdf[15,2]
plt_res = plt_res[:,:,5]

Plots.heatmap(1:size(plt_res,2),
        1:size(plt_res,1),
        plt_res,
    c=cgrad([:blue, :green, :yellow])
)


# plot basl resource type -------------------------------
pltm = mdf[5,3]
pltm1 = pltm[:,:,5]

# convert land cells to -1
pltm1[pltm1 .== -9999] .= -1

# plot
plt1 = Plots.heatmap(1:size(pltm1,2), 
              1:size(pltm1,1),
              pltm1,
              color = palette(:heat, length(unique(pltm1)))
)

plt1


# add fish movements ------------------------

# convert tuples to columns
plt_df = DataFrame(id = adf[:,2],
          x = getfield.(adf[:,3], 2),
          y = getfield.(adf[:,3], 1),
          type = adf[:,6]
)

# add fish to map
Plots.plot(
    plt1,
    plt_df[:,2],
    plt_df[:,3],
    color = repeat([:red, :blue, :green], inner = 5, outer = 201),
    group = plt_df[:,1],
    legend = false)


# basal resources through time -------------------
# plot basal resourse in a single cell through time

y = [mdf[i,2][5,10,1] for i in axes(mdf,1)]
x = 1:length(y)

Plots.plot(x, y)
# --------------------------------------------------







