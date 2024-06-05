

# add refugia from trout for both smelt and koaro - maybe then can get coexistence?

# change prey prefernce term, if term is 1 then species can ONLY feed on that chain

# run model as is - which ever species has resouerce pref = 1 goes extinct

# ---------------------------------------------------------------------------------------------------------------
# Brief overview ------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
#=
3D model of a lake food web. 

Overview:
1: generate a 3D lake environemnt from bathymetry data (currently csv)
2: define lake cell traits (e.g. resource type (pelagic vs littoral), resource growth rate)
3: populate lake with fish (3 species, trout (top predator), smelt and koaro (compeititors that consume resources produced in cells)
4: define fish movement (random in 3D, but weighted to move towards specific resource (koaro == littoral, smelt == pelagic), trout wherever koaro and smelt are)
5: define fish eat (koaro = littoral resource peference, smelt pelagic resource preference)
6: define fish reproduce (all three species have a chance of reporducing during a 2 month window each year, can't reproduce outside of this)
7: define fish die (high chance of death fter spawning for all 3 spcies, also if energy gets too low, also small random chance)
8: define cell behaviour - resource growth (follows discrete time logistic growth equation)
9: run model!
=#

#= 
To do
koaro/smelt predator avoidence - can do last
=#

#=
POM testing
1: fish combo scenarios 
2: relative abundance for each species/abundance time series?
3: pop size structure for each species?
=#

# ---------------------------------------------------------------------------------------------------------------
# load packages -------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

using Random
using Distributions
using StatsBase
using Agents, Agents.Pathfinding       # ABM
using InteractiveDynamics              # ABM
using GLMakie                          # for interactive plots
using Plots                            # standard plots
using CSV                              # importing bathymetry from csv
using DataFrames                       # importing bathymetry from csv

# ---------------------------------------------------------------------------------------------------------------
# define agents -------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------------------
# Model properties ----------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# somewhere to store params - not sure if this is correct setup, but I read somewhere if I am using mixed
# types I should use mutable struct
Base.@kwdef mutable struct Parameters
    swim_walkmap::BitArray
    #waterfinder::AStar
    can_swim::Vector{Tuple}
    heightmap2::Matrix{Int64}
	basal_resource::Array{Float64,3}
	basal_resource_type::Array{Int64,3}
	res_grow_r_lit::Float64
    res_grow_r_pel::Float64
    res_k_lit::Float64
    res_k_pel::Float64
    Δenergy_smelt::Float64
    Δenergy_koaro::Float64
    consume_amount_smelt::Float64
    consume_amount_koaro::Float64
    breed_prob_smelt::Float64     
    breed_prob_koaro::Float64
    breed_mortality_smelt::Float64
    breed_mortality_koaro::Float64 
    growth_rate_smelt::Float64
    growth_rate_koaro::Float64
    growth_rate_trout::Float64 
    length_mean_smelt::Float64
    length_sd_smelt::Float64
    vision_smelt::Int64
    vision_koaro::Int64
    size_maturity_smelt::Float64
    size_maturity_koaro::Float64
    mortality_random_smelt::Float64
    mortality_random_koaro::Float64
    resource_pref_adult_smelt::Float64
    resource_pref_juv_smelt::Float64
    resource_pref_adult_koaro::Float64
    resource_pref_juv_koaro::Float64
    prey_pref_trout::Float64
    fecundity_mean_smelt::Float64
    fecundity_sd_smelt::Float64
    fecundity_mean_koaro::Float64
    fecundity_sd_koaro::Float64
    Δenergy_trout::Float64
    vision_trout::Int64
    mortality_random_trout::Float64
    breed_prob_trout::Float64
    fecundity_mean_trout::Float64
    fecundity_sd_trout::Float64
    breed_mortality_trout::Float64
    length_mean_koaro::Float64
    length_sd_koaro::Float64
    length_mean_trout::Float64
    length_sd_trout::Float64
    tick::Int64
end

# ---------------------------------------------------------------------------------------------------------------
# initialize model ----------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# Currently place holder values used for parameters

function initialize_model(;
   # dims = (50, 50),                             # if we want ot use synthetic bathymetry
    lake_url::String = "data\\taupo_500m.csv",  # lake data
    max_littoral_depth::Int64 = 10,           # how far down does the littoral go in meters
    res_grow_r_lit::Float64 = 0.0001,              # cell resource growth rate - littoral resources - draw from distribution?
    res_grow_r_pel::Float64 = 0.0001,              # cell resource growth rate - pelagic resources - draw from distribution?
    res_k_lit::Float64 = 50.0,                  # cell resource carrying capacity - littoral - draw from distribution?
    res_k_pel::Float64 = 50.0,                   # cell resource carrying capacity - littoral - draw from distribution?
    n_smelt::Int64 = 50,                         # initial number of smelt
    n_koaro::Int64 = 0,                          # initial number of smelt
    n_trout::Int64 = 0,                          # initial number of smelt
    Δenergy_smelt::Float64 = 5.0,                # energy gained from eating resource - draw from distribution?
    Δenergy_koaro::Float64 = 5.0,
    consume_amount_smelt::Float64 = 10.0,        # max amount consumed in 1 timestep - draw from distribution?
    consume_amount_koaro::Float64 = 10.0,        # max amount consumed in 1 timestep - draw from distribution?
    breed_prob_smelt::Float64 = 0.01,            # probability of spawning (during seasonal window only) - draw from distribution?
    breed_prob_koaro::Float64 = 0.01,
    breed_mortality_smelt::Float64 = 1.0,        # probability of dying after spawning - draw from distribution?
    breed_mortality_koaro::Float64 = 1.0,        # probability of dying after spawning - draw from distribution?
    growth_rate_smelt::Float64 = 0.5,            # mm growth / time step - draw from distribution?
    growth_rate_koaro::Float64 = 0.2,
    growth_rate_trout::Float64 = 0.5,
    length_mean_smelt::Float64 = 25.0,           # mean adult smelt length - used for setting initial lengths
    length_sd_smelt::Float64 = 2.0,              # SD adult smelt length - used for setting initial lengths
    vision_smelt::Int64 = 1,                     # number of cells smelt can "see"
    vision_koaro::Int64 = 1,                     # number of cells koaro can "see"
    size_maturity_smelt::Float64 = 15.0,         # size (mm) when smelt transition from juvenile to adult
    size_maturity_koaro::Float64 = 50.0,         # size (mm) when smelt transition from juvenile to adult
    mortality_random_smelt::Float64 = 0.0,       # probability of random mortality each timestep
    mortality_random_koaro::Float64 = 0.0,       # probability of random mortality each timestep
    resource_pref_adult_smelt::Float64 = 0.99,   # adult smelt preference for (1) pelagic (0) littoral resources - but if koaro larvae present consume them
    resource_pref_juv_smelt::Float64 = 0.99,     # juvenile preference for pelagic (1) or littoral (0) resources
    resource_pref_adult_koaro::Float64 = 0.01,   # adult koaro preference for (1) pelagic (0) littoral resources 
    resource_pref_juv_koaro::Float64 = 0.99,     # juv koaro preference for (1) pelagic (0) littoral resources 
    prey_pref_trout::Float64 = 0.99,                     # trout preference for (0) koaro or (1) smelt
    fecundity_mean_smelt::Float64 = 50.0,         # mean number larvae produced
    fecundity_sd_smelt::Float64  = 1.0,           # SD number larvae produced
    fecundity_mean_koaro::Float64 = 5.0,
    fecundity_sd_koaro::Float64 = 1.0,
    Δenergy_trout::Float64 = 10.0,
    vision_trout::Int64 = 1,
    mortality_random_trout::Float64 = 0.0,
    breed_prob_trout::Float64 = 0.01,
    fecundity_mean_trout::Float64 = 2.0,
    fecundity_sd_trout::Float64 = 1.0,
    breed_mortality_trout::Float64 = 0.0,
    length_mean_koaro::Float64 = 50.0,
    length_sd_koaro::Float64 = 5.0,
    length_mean_trout::Float64 = 300.0,
    length_sd_trout::Float64 = 30.0,
    seed::Int64 = 23182,                       # rng seed
    )

    # define rng 
    rng = MersenneTwister(seed)

    # ----------------------------------------------------
    # create synthetic heatmap - alternative to using real lake topology

    #=
    heightmap2 = zeros(Int, dims)
    
    for i in 1:dims[1], j in 1:dims[2]
        if  (i + j) < dims[1] + 1
            heightmap2[i,j] = (min(i,j))
        else
            heightmap2[i,j] = min(dims[1] - i + 1, dims[2] - j + 1)
        end
    end

    =#

    
    # load lake topology ---------------------------------------
    lake_mtx = CSV.read(lake_url, DataFrame) |> Tables.matrix

    # convert to integer
    heightmap2 = floor.(Int, convert.(Float64, lake_mtx))::Matrix{Int64}
    heightmap2 .= heightmap2 .* -1
    # ----------------------------------------------------------- 

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
    lake_floor::Int64 = -1
    
    # lake dimensions ---------------------------------------------------------
    dims = (size(heightmap2)..., mx_dpth)
    space = GridSpace(dims, periodic = false)

    # 3d version of lake type, used for indexing
    basal_resource_type = repeat(lake_type, 1,1, dims[3])

    #  swimable space ----------------------------------------------------
    swim_walkmap = BitArray(falses(dims...))

    # fish can swim at any depth between the lake bed and lake suface
    for i in 1:dims[1], j in 1:dims[2]
        if lake_floor < heightmap2[i, j] <= mx_dpth
            swim_walkmap[i, j, (mx_dpth - heightmap2[i,j] + 1):mx_dpth] .= true
        end
    end

    # create vector of swimible cells - sampling from this will be quicker than using swim_walkmap later
    can_swim = Tuple.(findall(>(0), swim_walkmap))


    # create lake resource array --------------------------------------
    basal_resource = zeros(dims)

    # populate with basal resource - varies depending on if littoral or pelgic
    for i in 1:dims[1], j in 1:dims[2]
    if lake_type[i,j] == 0    # initial pelagic resource amount - 
        basal_resource[i, j, 1:mx_dpth] .= (res_k_pel / 2.0) 
    end
    if lake_type[i,j] == 1 # initial littoral  resource amount
        basal_resource[i, j, 1:mx_dpth] .= (res_k_lit / 2.0) 
    end
    end
  
    # make cells outside the simable area have 0 resources
    for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]
        if !swim_walkmap[i,j,k]
             basal_resource[i,j,k] = 0.0
         end
     end

    # define the properties -----------------------------------------
    properties = Parameters(
        swim_walkmap,
        #AStar(space; walkmap = swim_walkmap, diagonal_movement = true),
        can_swim,
        heightmap2,
        basal_resource,
        basal_resource_type,
        res_grow_r_lit,
        res_grow_r_pel,
        res_k_lit,
        res_k_pel,
        Δenergy_smelt,
        Δenergy_koaro,
        consume_amount_smelt,
        consume_amount_koaro,
        breed_prob_smelt,
        breed_prob_koaro,    
        breed_mortality_smelt,
        breed_mortality_koaro,
        growth_rate_smelt,
        growth_rate_koaro,
        growth_rate_trout,
        length_mean_smelt,
        length_sd_smelt, 
        vision_smelt,
        vision_koaro,
        size_maturity_smelt,
        size_maturity_koaro,
        mortality_random_smelt,
        mortality_random_koaro,
        resource_pref_adult_smelt,
        resource_pref_juv_smelt,
        resource_pref_adult_koaro,
        resource_pref_juv_koaro,
        prey_pref_trout,
        fecundity_mean_smelt,
        fecundity_sd_smelt,
        fecundity_mean_koaro,
        fecundity_sd_koaro,
        Δenergy_trout,
        vision_trout,
        mortality_random_trout,
        breed_prob_trout,
        fecundity_mean_trout,
        fecundity_sd_trout,
        breed_mortality_trout,
        length_mean_koaro,
        length_sd_koaro,
        length_mean_trout,
        length_sd_trout,
        1)

    # define model -----------------------------------
    model = ABM(Fish, space; properties, rng, scheduler = Schedulers.fastest, warn = true)




    # define a normal distribution - used to determine inital fish length for smelt
    #sm_sz = Normal(model.length_mean_smelt, model.length_sd_smelt)

    # Add agents -------------------------------------

     # add smelt only to pelagic at the start --------------------
        # storage
        swimable_cells = Vector{Tuple}()

        # check resources in current pos, if none, move
        # if model.basal_resource[smelt.pos...] < 10.0

            # get id of near cells - vision_range = range agent can "see"
            near_cells = can_swim

            # find which of the nearby cells are allowed to be moved into
            for cell in near_cells
                if swim_walkmap[cell...] > 0 && basal_resource_type[cell...] == 0
                    push!(swimable_cells, cell)
                end
            end 
    #----------------------------------------------------------------


    # Add smelt
    for _ in 1:n_smelt
                add_agent_pos!(
                    Smelt(
                        nextid(model),                               # Using `nextid` prevents us from having to manually keep track # of animal IDs
                        #sample(can_swim),
                        sample(swimable_cells), 
                        rand(model.rng, 10.0:100.0),                 # Fish starting energy level
                        #round(rand(sm_sz), digits = 3),             # initial length
                        45.0,                                        # initial length - TESTING
                        1),                                          # initial stage 
                    model,
                )
    end

        # add smelt only to pelagic at the start --------------------
                # storage
                swimable_cells2 = Vector{Tuple}()

                # check resources in current pos, if none, move
                # if model.basal_resource[smelt.pos...] < 10.0

                    # get id of near cells - vision_range = range agent can "see"
                    near_cells = can_swim

                    # find which of the nearby cells are allowed to be moved into
                    for cell in near_cells
                        if swim_walkmap[cell...] > 0 && basal_resource_type[cell...] == 1
                            push!(swimable_cells2, cell)
                        end
                    end 
        #----------------------------------------------------------------


    # Add koaro
    for _ in 1:n_koaro
        add_agent_pos!(
        Koaro(
            nextid(model),
            #sample(can_swim),
            sample(swimable_cells2),
            #random_walkable(model, model.waterfinder),  # random initial position
            rand(model.rng, 10.0:100.0),                 # Fish starting energy level
            #round(rand(sm_sz), digits = 3),             # initial length
            45.0,                                        # initial length - TESTING
            1),                                          # initial stage 
        model,
    )
    end

    # Add trout
    for _ in 1:n_trout
        add_agent_pos!(
        Trout(
            nextid(model), 
            sample(can_swim),                            
            #random_walkable(model, model.waterfinder),  # random initial position
            100.0,                                       # Fish starting energy level
            320.0,                                       # initial length
            1),                                          # initial stage 
        model,
    )
    end
    return model
end


# ---------------------------------------------------------------------------------------------------------------
# Fish movement wrapper - set up seperately for each species ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

function fish_step!(fish::Fish, model)
    if fish.type == :smelt
        smelt_step!(fish, model)
    elseif fish.type == :koaro
        koaro_step!(fish, model)
    elseif fish.type == :trout
        trout_step!(fish, model)
    end
end

# ---------------------------------------------------------------------------------------------------------------
# Smelt actions ------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

#=

# Smelt movement --------------------------------------------
function smelt_step!(smelt::Fish, model)
    
    swimable_cells = Vector{Tuple}()

    #=
    # find any juvenile koaro prey nearby
    koaro_prey = (
        x.pos for x in nearby_agents(smelt, model, model.vision_smelt) if
        x.type == :koaro && x.length < 5.0
        ) 
        
    # storage

     # if there are juvenile koaro prey nearby move to them - only if the smelt is an adult   
    if !isempty(koaro_prey) && smelt.length > model.size_maturity_smelt
    
        # need to add weight - move to cell with the most prey
        move_agent!(smelt, sample(collect(koaro_prey)), model)
   
     # if no prey koaro nearby, or smelt is juv look for cell resources
     # check if current pos has resources
     # if current cell has resources don't do anything if none, move
    
     #elseif model.basal_resource[smelt.pos...] < 2.0                         # TESTING VALUE  

    else
             =#

        # get id of near cells
        near_cells = nearby_positions(smelt.pos, model, model.vision_smelt)
         
        # find which of the nearby cells are allowed to be moved into
        for cell in near_cells
            if model.swim_walkmap[cell...] > 0 && model.basal_resource[cell...] > 2.0
                push!(swimable_cells, cell)
            end
        end 

        # if there is a neighbouring cell with resources > 2.0, randomly select one to move to,
        # weighted by resoruce preference
        if !isempty(swimable_cells)

            # First, determine if near cells are pel or lit and generate a vector of weights
            # get resource type for swimable_cells
            ind1 = [model.basal_resource_type[t...] for t in swimable_cells]::Vector{Int64}
            ind2 = convert(Array{Float64, 1}, ind1)::Vector{Float64}

             # pel and lit weights - depends on fish size
             if smelt.length < model.size_maturity_smelt
                pel_res_pref = model.resource_pref_juv_smelt
            else
                pel_res_pref = model.resource_pref_adult_smelt
            end
            
            # littoral pref
            lit_p = 1.0 - pel_res_pref

            # convert pel/lit indicies to weights
            ind2[ind2 .== 0.0] .= pel_res_pref
            ind2[ind2 .== 1.0] .= lit_p

            # randomly choose one of the nearby cells with resources and move
            # if all weights are 0, using weights is deterministic
            if sum(ind2) > 0.001
                move_agent!(smelt, sample(model.rng, swimable_cells, Weights(ind2)), model)
            else
                move_agent!(smelt, sample(model.rng, swimable_cells), model)
            end
        else

            # find which of the nearby cells are allowed to be moved onto
            for cell in near_cells
                if model.swim_walkmap[cell...] > 0
                    push!(swimable_cells, cell)
                end
            end
            move_agent!(smelt, sample(model.rng, swimable_cells), model)
        end
    #end 

    # smelt eating --------------------------------
    smelt_eat!(smelt, model)
    
     # Mortality -----------------------------------------
    # if there are very little resources smelt starve or random chance of death
    if smelt.energy < 1.0 || rand(model.rng) < model.mortality_random_smelt
        remove_agent!(smelt, model)
        return
    end

    # reproduction --------------------------------
    # testing  every 5 ticks - just for testing
   # if mod(model.tick, 10) == 0 && smelt.length > model.size_maturity_smelt
   #     smelt_reproduce!(smelt, model)
   # end
    
    # only chance of reproduction betwwen days 300-365 each year
    if smelt.length > model.size_maturity_smelt && rand(model.rng) < model.breed_prob_smelt 
        smelt_reproduce!(smelt, model)
     end


    # only chance of reproduction betwwen days 300-365 each year
    #if 300 < mod(model.tick, 365) < 365 && smelt.length > 30.0 && rand(model.rng) < model.breed_prob_smelt 
    #   smelt_reproduce!(smelt, model)
    #end

    # smelt loose energy after each step
    smelt.energy -= 5.0                                                # TESTING VALUE                          
    smelt.length += model.growth_rate_smelt                  # grow smelt
end


# smelt eat ------------------------------------------------
# eat koaro larvae, then cell resources 
function smelt_eat!(smelt::Fish, model)

     #=
    # check if there is koaro fry in current cell, if so eat them
    food = [x for x in agents_in_position(smelt, model) if x.type == :koaro && x.length < 0.0]
    #die_it = collect(food)
    
    # perference is to eat koaro fry, if not eat cell resource
    if !isempty(food)

        # determine how many koaro fry to eat (in one tick), increases with smelt size
        # if only a few fry presnt, eat all, otherwise scale with smelt length
        if length(food)  < round(smelt.length * 1.0)                                   #  TESTING VALUE
            n_eat = length(food)
        else
            n_eat = trunc(Int, smelt.length * 1.0)
        end

        # remove eaten koaro 
        remove_all!(model, StatsBase.sample(model.rng, food, n_eat, replace = false))

        # give smelt energy
        smelt.energy += (n_eat * model.Δenergy_smelt)

    # if there are resources available 
    elseif model.basal_resource[smelt.pos...] > 1.0     
        =#


    if model.basal_resource[smelt.pos...] > 1.0   
        # reduce resource amount - reduces resource by "consume_amount", unless less than that amount exists
        fut_res = model.basal_resource[smelt.pos...] - model.consume_amount_smelt * smelt.length
        
        if fut_res < 0.0 # if less than consume_amount exists  
            model.basal_resource[smelt.pos...] = 0.5
            smelt.energy += 0.5                             # give smelt energy
        else
            model.basal_resource[smelt.pos...] -= model.consume_amount_smelt * smelt.length
            smelt.energy += model.Δenergy_smelt
        end
    end
end


# smelt reproduction ---------------------------------------
function smelt_reproduce!(smelt::Fish, model)

    # how many juvs to add?
    sm_spwn = Normal(model.fecundity_mean_smelt, model.fecundity_sd_smelt)
    n_juv =  trunc(Int, rand(sm_spwn))
    
    #n_juv = 1::Int64  # TESTING

    # add fry
    for _ in 1:n_juv
        add_agent_pos!(
         Smelt(
             nextid(model),                                # Using `nextid` prevents us from having to manually keep track # of animal IDs
             sample(model.can_swim),                            
             #random_walkable(model, model.waterfinder),   # random starting location - could make pelagic
             rand(model.rng, 1:20),                        # Fish starting energy level - somewhere between 1 and 99
             2,                                            # initial length 2 mm
             0),                                           # initial stage juv
         model,
         )
    end

    # post spwaning mortality of adult
    if rand(model.rng) < model.breed_mortality_smelt
        remove_agent!(smelt, model)
    end
end

=#


function smelt_step!(smelt::Fish, model)

    # check if current pos has resources, - will need to update to check if predator is present 
    # if current cell has resources (and no predator) don't do anything 
    
   # storage
   swimable_cells = Vector{Tuple}()

    # check resources in current pos, if none, move
    # if model.basal_resource[smelt.pos...] < 10.0

        # get id of near cells - vision_range = range agent can "see"
        near_cells = nearby_positions(smelt.pos, model, model.vision_koaro)

        # find which of the nearby cells are allowed to be moved into
        for cell in near_cells
            if model.swim_walkmap[cell...] > 0 && model.basal_resource[cell...] > 2.0
                push!(swimable_cells, cell)
            end
        end 

        # if there is a neighbouring cell with resources > 0, randomly select one to move to,
        # weighted by resoruce preference
        if !isempty(swimable_cells)

            # First, determine if near cells are pel or lit and generate a vector of weights
            
            # get resource type for swimable_cells
            ind1 = [model.basal_resource_type[t...] for t in swimable_cells]::Vector{Int64}
            ind2 = convert(Array{Float64, 1}, ind1)::Vector{Float64}

            # pel and lit weights - depends on fish size
            if smelt.length < 10.0
                pel_res_pref = model.resource_pref_juv_smelt
            else
                pel_res_pref = model.resource_pref_adult_smelt
            end

            # littoral weight
            lit_p = 1.0 - pel_res_pref

            # convert pel/lit indicies to weights
            ind2[ind2 .== 0.0] .= pel_res_pref
            ind2[ind2 .== 1.0] .= lit_p

            # randomly choose one of the nearby cells with resources and move
            # if all weights are 0, using weights is deterministic
            if sum(ind2) > 0.001
                move_agent!(smelt, sample(model.rng, swimable_cells, Weights(ind2)), model)
            else
                move_agent!(smelt, sample(model.rng, swimable_cells), model)
            end
        else # if no neighbouring cells have food, pick one at random

            # find which of the nearby cells are allowed to be moved onto
            for cell in near_cells
                if model.swim_walkmap[cell...] > 0
                    push!(swimable_cells, cell)
                end
            end
            move_agent!(smelt, sample(swimable_cells), model)
        end
    #end 
    

    # smelt eating - see function below
    koaro_eat!(smelt, model)

    #  if < 1.0 die or small random chance
    if smelt.energy < 1.0 || rand(model.rng) < model.mortality_random_koaro
        remove_agent!(smelt, model)
        return
    end
    
    # reproduction --------------------------------
    # testing  every 5 ticks - just for testing
   # if mod(model.tick, 10) == 0 && smelt.length > model.size_maturity_koaro 
   #     kaoro_reproduce!(smelt, model)
   # end
   
    if smelt.length > model.size_maturity_koaro && rand(model.rng) < model.breed_prob_koaro 
        kaoro_reproduce!(smelt, model)
    end

    # in reality it will need to be between days 300 - 365
    #if 300 < mod(model.tick, 365) < 365 && smelt.length > 40.0 && rand(model.rng) < model.breed_prob_koaro 
    #    kaoro_reproduce!(smelt, model)
    #end
    
      # smelt loose energy after each step
 #     smelt.energy -= 5.0                           # TESTING VALUE
      smelt.length += model.growth_rate_koaro                                   # grow smelt
end


# smelt eat - just cell resources---------------------------
function smelt_eat!(smelt::Fish, model)

    if model.basal_resource[smelt.pos...] > 1.0     # if there are resources available 
        
        # reduce resource amount - reduces resource by "consume_amount", unless less than that amount exists, then consume all
        fut_res = model.basal_resource[smelt.pos...] - model.consume_amount_koaro  * smelt.length

        if fut_res < 0.0 
            model.basal_resource[smelt.pos...] = 0.5
            smelt.energy += 0.5                         # give smelt energy
        else
            model.basal_resource[smelt.pos...] -= model.consume_amount_koaro * smelt.length
            smelt.energy += model.Δenergy_koaro                     # give smelt energy            
        end
    end
end


# smelt reproduction---------------------------------------
function smelt_reproduce!(smelt::Fish, model)

    # how many juvs to add?
    fecund_koaro = Normal(model.fecundity_mean_koaro, model.fecundity_sd_koaro)
    n_juv =  trunc(Int, rand(fecund_koaro))
    
    #n_juv::Int64 = 1                      # TESTING

    for _ in 1:n_juv
        add_agent_pos!(
         Smelt(
             nextid(model),                               # Using `nextid` prevents us from having to manually keep track # of animal IDs
             sample(model.can_swim),                            
             #random_walkable(model, model.waterfinder),
             rand(model.rng, 50:100) - 1,                   # Fish starting energy level - somewhere between 1 and 99
             2,                                           # initial length 2 mm
             0),                                          # initial stage juv
         model,
     )
    end

    # post spwaning mortality of adult
    if rand(model.rng) < model.breed_mortality_koaro
        remove_agent!(smelt, model)
    end
end




# ---------------------------------------------------------------------------------------------------------------
#  koaro actions ------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# koaro movement --------------------------------------------
function koaro_step!(koaro::Fish, model)

    # check if current pos has resources, - will need to update to check if predator is present 
    # if current cell has resources (and no predator) don't do anything 
    
   # storage
   swimable_cells = Vector{Tuple}()

    # check resources in current pos, if none, move
    # if model.basal_resource[koaro.pos...] < 10.0

        # get id of near cells - vision_range = range agent can "see"
        near_cells = nearby_positions(koaro.pos, model, model.vision_koaro)

        # find which of the nearby cells are allowed to be moved into
        for cell in near_cells
            if model.swim_walkmap[cell...] > 0 && model.basal_resource[cell...] > 2.0
                push!(swimable_cells, cell)
            end
        end 

        # if there is a neighbouring cell with resources > 0, randomly select one to move to,
        # weighted by resoruce preference
        if !isempty(swimable_cells)

            # First, determine if near cells are pel or lit and generate a vector of weights
            
            # get resource type for swimable_cells
            ind1 = [model.basal_resource_type[t...] for t in swimable_cells]::Vector{Int64}
            ind2 = convert(Array{Float64, 1}, ind1)::Vector{Float64}

            # pel and lit weights - depends on fish size
            if koaro.length < 10.0
                pel_res_pref = model.resource_pref_juv_koaro
            else
                pel_res_pref = model.resource_pref_adult_koaro
            end

            # littoral weight
            lit_p = 1.0 - pel_res_pref

            # convert pel/lit indicies to weights
            ind2[ind2 .== 0.0] .= pel_res_pref
            ind2[ind2 .== 1.0] .= lit_p

            # randomly choose one of the nearby cells with resources and move
            # if all weights are 0, using weights is deterministic
            if sum(ind2) > 0.001
                move_agent!(koaro, sample(model.rng, swimable_cells, Weights(ind2)), model)
            else
                move_agent!(koaro, sample(model.rng, swimable_cells), model)
            end
        else
            
            # find which of the nearby cells are allowed to be moved onto
            for cell in near_cells
                if model.swim_walkmap[cell...] > 0
                    push!(swimable_cells, cell)
                end
            end
            
            move_agent!(koaro, sample(swimable_cells), model)
        end
    #end 
    

    # koaro eating - see function below
    koaro_eat!(koaro, model)

    #  if < 1.0 die or small random chance
    if koaro.energy < 1.0 || rand(model.rng) < model.mortality_random_koaro
        remove_agent!(koaro, model)
        return
    end
    
    # reproduction --------------------------------
    # testing  every 5 ticks - just for testing
    # if mod(model.tick, 10) == 0 && koaro.length > model.size_maturity_koaro 
    #     kaoro_reproduce!(koaro, model)
    # end
   
    if koaro.length > model.size_maturity_koaro && rand(model.rng) < model.breed_prob_koaro 
        kaoro_reproduce!(koaro, model)
    end

    # in reality it will need to be between days 300 - 365
    #if 300 < mod(model.tick, 365) < 365 && koaro.length > 40.0 && rand(model.rng) < model.breed_prob_koaro 
    #    kaoro_reproduce!(koaro, model)
    #end
    
      # koaro loose energy after each step
     # koaro.energy -= 5.0                           # TESTING VALUE
      koaro.length += model.growth_rate_koaro                                   # grow koaro
end


# koaro eat - just cell resources---------------------------
function koaro_eat!(koaro::Fish, model)

    if model.basal_resource[koaro.pos...] > 1.0     # if there are resources available 
        
        # reduce resource amount - reduces resource by "consume_amount", unless less than that amount exists, then consume all
        fut_res = model.basal_resource[koaro.pos...] - model.consume_amount_koaro  * koaro.length

        if fut_res < 0.0 
            model.basal_resource[koaro.pos...] = 0.5
            koaro.energy += 0.5                         # give koaro energy
        else
            model.basal_resource[koaro.pos...] -= model.consume_amount_koaro * koaro.length
            koaro.energy += model.Δenergy_koaro                     # give koaro energy            
        end
    end
end


# koaro reproduction---------------------------------------
function kaoro_reproduce!(koaro::Fish, model)

    # how many juvs to add?
    fecund_koaro = Normal(model.fecundity_mean_koaro, model.fecundity_sd_koaro)
    n_juv =  trunc(Int, rand(fecund_koaro))
    
    #n_juv::Int64 = 1                      # TESTING

    for _ in 1:n_juv
        add_agent_pos!(
         Koaro(
             nextid(model),                               # Using `nextid` prevents us from having to manually keep track # of animal IDs
             sample(model.can_swim),                            
             #random_walkable(model, model.waterfinder),
             rand(model.rng, 50:100) - 1,                   # Fish starting energy level - somewhere between 1 and 99
             2,                                           # initial length 2 mm
             0),                                          # initial stage juv
         model,
     )
    end

    # post spwaning mortality of adult
    if rand(model.rng) < model.breed_mortality_koaro
        remove_agent!(koaro, model)
    end
end

# ---------------------------------------------------------------------------------------------------------------
#  trout actions ------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# trout movement -------------------------
function trout_step!(trout::Fish, model)   

    # find any prey nearby  
    prey = (x for x in nearby_agents(trout,
    initialised_model, model.vision_trout) if x.type == :koaro || x.type == :smelt)

     # if there are prey nearby move to them 
    if !isempty(prey)
    
        # create a vector of weights based on prey preference
        p_type = [x.type for x in prey]::Vector{Symbol}
        pref_t = Vector{Float64}()
                
        for p in p_type
            if p == :smelt
                push!(pref_t, model.prey_pref_trout)
            else
                push!(pref_t, 1.0 - model.prey_pref_trout)
            end
        end
        
        # postition of prey
        prey_pos = (x.pos for x in prey)

        # move
        move_agent!(trout, sample(collect(prey_pos), Weights(pref_t)), model)
      
    else
        # if none of the near cells have prey, just pick one at random
        near_cells = nearby_positions(trout.pos, model, model.vision_trout)
        
        # storage
        swimable_cells = Vector{Tuple}()
        
        # find which of the nearby cells are allowed to be moved into
        for cell in near_cells
            if model.swim_walkmap[cell...] > 0
                push!(swimable_cells, cell)
            end
        end
        
        move_agent!(trout, sample(swimable_cells), model)
        
    end

    # trout eating --------------------------------
    trout_eat!(trout, model)

  # Mortality -----------------------------------------
    # if there are very little resources trout starve or random chance of death
    if trout.energy < 1.0 || rand(model.rng) < model.mortality_random_trout
        remove_agent!(trout, model)
        return
    end
    
    # reproduction --------------------------------
    # testing  every 5 ticks - just for testing
    # if mod(model.tick, 10) == 0 && trout.length > 300.0 
    #     reproduce_trout!(trout, model)
    # end
    
    if trout.length > 100.0 && rand(model.rng) < model.breed_prob_trout 
        reproduce_trout!(trout, model)
    end

    # in reality it will need to be between days 300 - 365
    #if 300 < mod(model.tick, 365) < 365 && trout.length > 350.0 && rand(model.rng) < model.breed_prob_trout 
    #    reproduce_trout!(trout, model)
    #end

    # trout loose energy after each step
    trout.length += model.growth_rate_trout
   # trout.energy -= 5.0                                                    # TESTING VALUE

  
end


# trout eat -----------------------------
function trout_eat!(trout::Fish, model)

    # check if there is koaro fry in current cell, if so eat them
    food = [x for x in agents_in_position(trout, model) if x.type == :koaro || x.type == :smelt]
    die_it = collect(food)
    
    if !isempty(die_it)

        n_eat::Int64 = 1                #  TESTING VALUE

        #=
        # determine how many prey to eat, increases with trout size
        # if only a few prey present eat all, otherwise scale with trout length
        if length(die_it)  < trunc(Int, trout.length * 1.0)                                  
            #n_eat = length(die_it)
        else
            #n_eat = trunc(Int, trout.length * 1.0)
        end
        =#

        # remove eaten prey 
        remove_all!(model, StatsBase.sample(model.rng, die_it, n_eat, replace = false))

        # give trout energy
        trout.energy += (n_eat * model.Δenergy_trout)

    end

end


# trout reproduction ---------------------
function reproduce_trout!(trout::Fish, model)

    # how many juvs to add?
    fecund_trout = Normal(model.fecundity_mean_trout, model.fecundity_sd_trout)
    n_juv = trunc(Int, rand(fecund_trout))
    
   # n_juv::Int64 = 1                      # TESTING

    for _ in 1:n_juv
        add_agent_pos!(
         Trout(
             nextid(model),                               # Using `nextid` prevents us from having to manually keep track # of animal IDs
             sample(model.can_swim),                            
            # random_walkable(model, model.waterfinder),
             rand(model.rng, 10:50),                   # Fish starting energy level - somewhere between 1 and 20
             300,                                           # initial length 2 mm
             0),                                          # initial stage juv
         model,
     )
    end
    
    # post spwaning mortality of adult
    if rand(model.rng) < model.breed_mortality_trout
        remove_agent!(trout, model)
    end
end


# ---------------------------------------------------------------------------------------------------------------
# cell resource growth  -----------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# follows discrete time logistic growth
function resource_growth(model)

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


# ---------------------------------------------------------------------------
# plotting summay lines through time -----------------------------------------
# ---------------------------------------------------------------------------

initialised_model = initialize_model(
   # dims = (50, 50),                    # landscape size
    lake_url = "data\\taupo_500m.csv",  # lake data
    max_littoral_depth = 50,            # basal resources  
    res_k_lit = 100.0,
    res_k_pel = 100.0,                
    res_grow_r_lit = 0.15,
    res_grow_r_pel = 0.15,
    n_smelt = 50,                      # initial fish abund
    n_koaro = 50,
    n_trout = 0,
    vision_smelt = 1,                   # vision
    vision_koaro = 1,
    vision_trout = 1,
    resource_pref_adult_smelt = 1.0,   # prey preference
    resource_pref_juv_smelt = 1.0,
    resource_pref_adult_koaro = 0.0,
    resource_pref_juv_koaro = 0.0,
    prey_pref_trout = 0.95,
    consume_amount_smelt = 1.0,        # resource consumption
    consume_amount_koaro = 1.0,
    Δenergy_smelt = 10.0,               # energy gained
    Δenergy_koaro = 10.0,
    Δenergy_trout = 20.0,
    length_mean_smelt = 25.0,           # length
    length_sd_smelt = 0.0,
    length_mean_koaro = 25.0,
    length_sd_koaro = 0.0,
    length_mean_trout = 300.0,
    length_sd_trout = 0.0,
    growth_rate_smelt = 1.0,            # growth rate (mm/timestep)
    growth_rate_koaro = 1.0,
    growth_rate_trout = 1.0,
    size_maturity_smelt = 20.0,         # adult size
    size_maturity_koaro = 20.0,
    #size_maturity_trout = 300.0,
    breed_prob_smelt = 0.003,            # breeding prob
    breed_prob_koaro = 0.003,
    breed_prob_trout = 0.003,
    fecundity_mean_smelt = 10.0,         # fecundity
    fecundity_sd_smelt  = 0.0,
    fecundity_mean_koaro = 10.0,
    fecundity_sd_koaro = 0.0,
    fecundity_mean_trout = 10.0,
    fecundity_sd_trout = 0.0,
    breed_mortality_smelt = 0.0,        # mortality spawning
    breed_mortality_koaro = 0.0,
    breed_mortality_trout = 0.0,
    mortality_random_smelt = 0.0,      # mortality random
    mortality_random_koaro = 0.0,
    mortality_random_trout = 0.0,
    seed = trunc(Int, rand() * 10000)
)



 function plot_lines(initialised_model, steps)
    
    # define what I want to gather data on
    smelt(a) = a.type == :smelt
    koaro(a) = a.type == :koaro
    trout(a) = a.type == :trout
    energy(a) = a.energy
    mean_res(model) = mean(model.basal_resource)
    
    # summary stats ()
    adata = [(smelt, count), (koaro, count), (trout, count)]
    mdata = [mean_res]
    
    adf, mdf = run!(initialised_model, fish_step!, resource_growth, steps; adata, mdata)
    
    adf
    mdf
    
    


    # plot
    p1 = Plots.plot(adf[:,1], 
    [adf[:,2] adf[:,3] adf[:,4]],
               label=["smelt" "koaro" "trout"]
    )

    p2 = Plots.plot(adf[:,1],
    [mdf[:,2]],
    label=["resource"])

    Plots.plot(p1, p2, layout=(2,1))



end







initialised_model = initialize_model(
    #dims = (50, 50),                    # landscape size
    max_littoral_depth = 50,            # basal resources  
    res_k_lit = 100.0,
    res_k_pel = 100.0,                
    res_grow_r_lit = 0.15,
    res_grow_r_pel = 0.15,
    n_smelt = 50,                      # initial fish abund
    n_koaro = 50,
    n_trout = 0,
    vision_smelt = 1,                   # vision
    vision_koaro = 1,
    vision_trout = 1,
    resource_pref_adult_smelt = 1.0,   # prey preference
    resource_pref_juv_smelt = 1.0,
    resource_pref_adult_koaro = 0.0,
    resource_pref_juv_koaro = 0.0,
    prey_pref_trout = 0.95,
    consume_amount_smelt = 1.0,        # resource consumption
    consume_amount_koaro = 1.0,
    Δenergy_smelt = 10.0,               # energy gained
    Δenergy_koaro = 10.0,
    Δenergy_trout = 20.0,
    length_mean_smelt = 25.0,           # length
    length_sd_smelt = 0.0,
    length_mean_koaro = 25.0,
    length_sd_koaro = 0.0,
    length_mean_trout = 300.0,
    length_sd_trout = 0.0,
    growth_rate_smelt = 1.0,            # growth rate (mm/timestep)
    growth_rate_koaro = 1.0,
    growth_rate_trout = 1.0,
    size_maturity_smelt = 20.0,         # adult size
    size_maturity_koaro = 20.0,
    #size_maturity_trout = 300.0,
    breed_prob_smelt = 0.003,            # breeding prob
    breed_prob_koaro = 0.003,
    breed_prob_trout = 0.003,
    fecundity_mean_smelt = 10.0,         # fecundity
    fecundity_sd_smelt  = 0.0,
    fecundity_mean_koaro = 10.0,
    fecundity_sd_koaro = 0.0,
    fecundity_mean_trout = 10.0,
    fecundity_sd_trout = 0.0,
    breed_mortality_smelt = 0.0,        # mortality spawning
    breed_mortality_koaro = 0.0,
    breed_mortality_trout = 0.0,
    mortality_random_smelt = 0.0,      # mortality random
    mortality_random_koaro = 0.0,
    mortality_random_trout = 0.0,
    seed = trunc(Int, rand() * 10000)
)



adf

steps = 10

    
    # define what I want to gather data on
    smelt(a) = a.type == :smelt
    koaro(a) = a.type == :koaro
    trout(a) = a.type == :trout
    energy(a) = a.energy
    
    # summary stats ()
    adata = [:type, :pos]
    mdata = [:basal_resource_type]
    
    adf, mdf = run!(initialised_model, fish_step!, resource_growth, steps; adata, mdata)
    
    adf
    

    using HDF5
    h5write("mdf.h5", "mygroup", mdf[1, 2])

    mdf[1, 2][:,:,1]

    Plots.heatmap(1:64,
        1:66,
        mdf[1, 2][:,:,1],
        c=cgrad([:blue,:brown]),
        title="My title")


        Plots.heatmap(1:64,
        1:66,
        mdf[1, 2][:,:,1],
        colour = ([:blue,:brown, :green]),
        title="My title")



using DataFramesMeta

   # number of species each time step
   df_count = combine(groupby(adf, [:step, :type]), nrow => :count)
   df_count = unstack(df_count, :type, :count)
   df_count = df_count[:, 2:3]

  p1 = Plots.plot(Matrix(df_count),
   labels = permutedims(names(df_count)),
   legend = :topleft)




   # plot resuource through time
    p2 = Plots.plot(mdf[:,1],
    [mdf[:,2]],
    label=["resource"])










    # note resource will be roughly 1/3 where it should be, becaus it is averaged over non-accessable (0.0) cells
plot_lines(initialised_model, 1000)


# ---------------------------------------------------------------------------
# Raw data results ----------------------------------------------------------
# ---------------------------------------------------------------------------

initialised_model = initialize_model(prey_pref_trout = 0.99,
                                    n_smelt = 500,
                                    n_koaro = 0,
                                    n_trout = 0,
                                    vision_trout = 1,
                                    vision_smelt = 1,
                                    vision_koaro = 1,
                                    res_grow_r_lit = 0.005,
                                    res_grow_r_pel = 0.005)

# somewhere to store results
adata = [:pos, :energy, :length, :type]
mdata = [:basal_resource, :basal_resource_type, :tick]

steps = 500

adf, mdf = run!(initialised_model, fish_step!, resource_growth, steps; adata, mdata)

adf

# convert tuple to columns 
adf2 = transform(adf, :pos => AsTable)

CSV.write("movement.csv", adf2)





adf[adf.type .== :trout, :]


mdf
min(mdf[800, 2][:,:,1]...)

mdf[200, 2][:,:,1]


# -----------------------------------------------------------------------------------------------------
# Plot movement map -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

initialised_model = initialize_model(
    #dims = (5, 5),                    # landscape size
    max_littoral_depth = 100,            # basal resources  
    res_k_lit = 50.0,
    res_k_pel = 50.0,                
    res_grow_r_lit = 0.15,
    res_grow_r_pel = 0.15,
    n_smelt = 20,                      # initial fish abund
    n_koaro = 20,
    n_trout = 5,
    vision_smelt = 1,                   # vision
    vision_koaro = 1,
    vision_trout = 2,
    resource_pref_adult_smelt = 0.9,   # prey preference
    resource_pref_juv_smelt = 0.9,
    resource_pref_adult_koaro = 0.1,
    resource_pref_juv_koaro = 0.1,
    prey_pref_trout = 0.8,
    consume_amount_smelt = 10.0,        # resource consumption
    consume_amount_koaro = 10.0,
    Δenergy_smelt = 10.0,               # energy gained
    Δenergy_koaro = 10.0,
    Δenergy_trout = 50.0,
    length_mean_smelt = 25.0,           # length
    length_sd_smelt = 2.0,
    #length_mean_koaro = 25.0,
    #length_sd_koaro = 2.0,
    #length_mean_trout = 25.0,
    #length_sd_trout = 2.0,
    growth_rate_smelt = 0.5,            # growth rate (mm/timestep)
    growth_rate_koaro = 0.2,
    growth_rate_trout = 0.5,
    size_maturity_smelt = 15.0,         # adult size
    size_maturity_koaro = 10.0,
    #size_maturity_trout
    breed_prob_smelt = 0.0,            # breeding prob
    breed_prob_koaro = 0.0,
    breed_prob_trout = 0.0,
    fecundity_mean_smelt = 2.0,         # fecundity
    fecundity_sd_smelt  = 1.0,
    fecundity_mean_koaro = 2.0,
    fecundity_sd_koaro = 1.0,
    fecundity_mean_trout = 2.0,
    fecundity_sd_trout = 1.0,
    breed_mortality_smelt = 0.0,        # mortality spawning
    breed_mortality_koaro = 0.7,
    breed_mortality_trout = 0.0,
    mortality_random_smelt = 0.0,      # mortality random
    mortality_random_koaro = 0.0,
    mortality_random_trout = 0.0
)

function test_plot(initialised_model, steps = 100, n_fish_plot = 50)


    # somewhere to store results
    adata = [:pos, :energy, :length, :type]
    mdata = [:basal_resource, :basal_resource_type, :tick]

    # obtainer = copy - use this if you need to update the mdf output - by default if the output is mutable container it 
    # won't show updates. using obtainer = copy will reduce performance, only use for prototyping 
    adf, mdf = run!(initialised_model, fish_step!, resource_growth, steps; adata, mdata, obtainer = deepcopy)


    # plot basl resource at a random depth -----------------
    
    step_r = steps
    depth = 1
    plt_res = mdf[step_r,2]
    plt_res = plt_res[:,:,depth]

    Plots.heatmap(1:size(plt_res,2),
            1:size(plt_res,1),
            plt_res,
        c=cgrad([:blue, :green, :yellow])
    )


    # plot basl resource type -------------------------------
    pltm = mdf[1,3]
    pltm1 = pltm[:,:, 5]

    # convert land cells to -1
    pltm1[pltm1 .== -9999] .= -1

    # plot
    plt1 = Plots.heatmap(1:size(pltm1,2), 
                1:size(pltm1,1),
                pltm1,
                color = palette(:heat, length(unique(pltm1)))
    )

    # add fish movements ------------------------
    adf
    # convert tuples to columns
    plt_df = DataFrame(id = adf[:,2],
            x = getfield.(adf[:,3], 2),
            y = getfield.(adf[:,3], 1),
            type = adf[:,6]
    )

    # add colour column
    plt_df.col = @. ifelse(plt_df.type == :trout, :blue, ifelse(plt_df.type == :smelt, :green, :red) )

   # sample a few fish to plot
   kp_fish  = sample(plt_df[:,1], n_fish_plot)
   plt_df_sub = plt_df[in.(plt_df.id, Ref(kp_fish)), :]

    # add fish to map
    Plots.plot(
        plt1,
        plt_df_sub[:,2],
        plt_df_sub[:,3],
        color = plt_df_sub[:,5],
        group = plt_df_sub[:,1],
        legend = false)

end


# green = smelt
# red = koaro
# blue = trout
test_plot(initialised_model)














# -----------------------------------------------------------------------------------------------------
# interactive plot ------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

initialised_model = initialize_model(prey_pref_trout = 0.99,
                                    n_smelt = 1,
                                    n_koaro = 0,
                                    n_trout = 0
                                    )

# agnt colours
animalcolor(a) =
if a.type == :trout
        :yellow
elseif a.type == :koaro
        :red
elseif a.type == :smelt
        :blue
end

# plot bits
plotkwargs = (;
    ac = animalcolor,
    as = 2,
    am = :circle,
)


# parameters to have on sliders
params = Dict(
    :mortality_random_trout => 0:0.01:1,
    :resource_pref_adult_smelt => 0:0.01:1,
    :resource_pref_juv_smelt => 0:0.01:1,
    :resource_pref_adult_koaro => 0:0.01:1,
    :resource_pref_juv_koaro => 0:0.01:1,
    :prey_pref_trout  => 0:0.01:1,
)

# forces GLMakie to open a seperate plot window, rather than using 
# the VSCode plotting window ... needed for interactive plots after recent GLMakie update
GLMakie.activate!(inline = false)

fig, ax, abmobs = abmplot(initialised_model;
    agent_step! = fish_step!,
    model_step! = resource_growth,
    params,
    plotkwargs...)
fig

