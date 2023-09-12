

# scenarios:
# 1: Smelt and trout coexist - DONE
# 2: all three - DONE
# 3: koaro alone - DONE
# 4: smelt alone - DONE

# run the model, spit out df
function run_and_save(initialised_model)

    #function run_and_save()
    # define what I want to gather data on
    smelt(a) = a.type == :smelt
    koaro(a) = a.type == :koaro
    trout(a) = a.type == :trout
    mean_res(model) = mean(model.basal_resource)
    
    # summary stats
    adata = [(smelt, count), (koaro, count), (trout, count)]
    mdata = [mean_res]
    steps = 1000

    adf, mdf = run!(initialised_model, fish_step!, resource_growth, steps; adata, mdata)

    df_res  = outerjoin(adf, mdf, on=:step)
    return df_res
end




# trout and smelt coextistence -------------
for i in 1:10 

    initialised_model = initialize_model(
        dims = (50, 50),                    # landscape size
        max_littoral_depth = 20,            # basal resources  
        res_k_lit = 100.0,
        res_k_pel = 100.0,                
        res_grow_r_lit = 0.15,
        res_grow_r_pel = 0.15,
        n_smelt = 1000,                      # initial fish abund
        n_koaro = 0,
        n_trout = 100,
        vision_smelt = 1,                   # vision
        vision_koaro = 1,
        vision_trout = 1,
        resource_pref_adult_smelt = 0.5,   # prey preference
        resource_pref_juv_smelt = 1.0,
        resource_pref_adult_koaro = 1.0,
        resource_pref_juv_koaro = 1.0,
        prey_pref_trout = 0.95,
        consume_amount_smelt = 1.0,        # resource consumption
        consume_amount_koaro = 1.0,
        Δenergy_smelt = 10.0,               # energy gained
        Δenergy_koaro = 10.0,
        Δenergy_trout = 10.0,
        Δenergy_loss_smelt = 10.0,          # energy lost              
        Δenergy_loss_koaro = 1.0,
        Δenergy_loss_trout = 1.0,          
        length_mean_smelt = 25.0,           # length
        length_sd_smelt = 0.0,
        length_mean_koaro = 25.0,
        length_sd_koaro = 0.0,
        length_mean_trout = 25.0,
        length_sd_trout = 0.0,
        growth_rate_smelt = 1.0,            # growth rate (mm/timestep)
        growth_rate_koaro = 1.0,
        growth_rate_trout = 1.0,
        size_maturity_smelt = 20.0,         # adult size
        size_maturity_koaro = 20.0,
        size_maturity_trout = 20.0,
        breed_prob_smelt = 0.003,            # breeding prob
        breed_prob_koaro = 0.003,
        breed_prob_trout = 0.003,
        fecundity_mean_smelt = 10.0,         # fecundity
        fecundity_sd_smelt  = 0.0,
        fecundity_mean_koaro = 10.0,
        fecundity_sd_koaro = 0.0,
        fecundity_mean_trout = 5.0,
        fecundity_sd_trout = 0.0,
        breed_mortality_smelt = 0.0,        # mortality spawning
        breed_mortality_koaro = 0.0,
        breed_mortality_trout = 0.0,
        mortality_random_smelt = 0.0,      # mortality random
        mortality_random_koaro = 0.0,
        mortality_random_trout = 0.0,
        seed = trunc(Int, rand() * 10000)
)
    res = run_and_save(initialised_model)
    res.exp .= 1
    res.rep .= i
    res = select!(res, Not([:count_koaro]))

    CSV.write(string("data_created\\exp1_smelt_and_trout", i, ".csv"), res)
end
       

# all three - sort of works ----------------------------
for i in 1:10

    initialised_model = initialize_model(
        dims = (50, 50),                    # landscape size
        max_littoral_depth = 30,            # basal resources  
        res_k_lit = 100.0,
        res_k_pel = 100.0,                
        res_grow_r_lit = 0.2,
        res_grow_r_pel = 0.2,
        n_smelt = 1000,                      # initial fish abund
        n_koaro = 500,
        n_trout = 100,
        vision_smelt = 1,                   # vision
        vision_koaro = 1,
        vision_trout = 1,
        resource_pref_adult_smelt = 0.5,   # prey preference
        resource_pref_juv_smelt = 1.0,
        resource_pref_adult_koaro = 0.9,
        resource_pref_juv_koaro = 1.0,
        prey_pref_trout = 1.0,
        consume_amount_smelt = 1.0,        # resource consumption
        consume_amount_koaro = 1.0,
        Δenergy_smelt = 10.0,               # energy gained
        Δenergy_koaro = 10.0,
        Δenergy_trout = 10.0,
        Δenergy_loss_smelt = 10.0,          # energy lost              
        Δenergy_loss_koaro = 10.0,
        Δenergy_loss_trout = 1.0,          
        length_mean_smelt = 25.0,           # length
        length_sd_smelt = 0.0,
        length_mean_koaro = 25.0,
        length_sd_koaro = 0.0,
        length_mean_trout = 25.0,
        length_sd_trout = 0.0,
        growth_rate_smelt = 1.0,            # growth rate (mm/timestep)
        growth_rate_koaro = 1.0,
        growth_rate_trout = 1.0,
        size_maturity_smelt = 20.0,         # adult size
        size_maturity_koaro = 20.0,
        size_maturity_trout = 20.0,
        breed_prob_smelt = 0.003,            # breeding prob
        breed_prob_koaro = 0.003,
        breed_prob_trout = 0.003,
        fecundity_mean_smelt = 15.0,         # fecundity
        fecundity_sd_smelt  = 0.0,
        fecundity_mean_koaro = 15.0,
        fecundity_sd_koaro = 0.0,
        fecundity_mean_trout = 5.0,
        fecundity_sd_trout = 0.0,
        breed_mortality_smelt = 0.0,        # mortality spawning
        breed_mortality_koaro = 0.0,
        breed_mortality_trout = 0.0,
        mortality_random_smelt = 0.001,      # mortality random 
        mortality_random_koaro = 0.001,
        mortality_random_trout = 0.0,
        seed = trunc(Int, rand() * 10000)
    )

    res = run_and_save(initialised_model)
    res.exp .= 2
    res.rep .= i

    CSV.write(string("data_created\\exp2_all_three", i, ".csv"), res)
end
       

# koaro alone -----------------------------------------------
for i in 1:10 

    initialised_model = initialize_model(
        dims = (50, 50),                    # landscape size
        max_littoral_depth = 30,            # basal resources  
        res_k_lit = 100.0,
        res_k_pel = 100.0,                
        res_grow_r_lit = 0.2,
        res_grow_r_pel = 0.2,
        n_smelt = 0,                      # initial fish abund
        n_koaro = 500,
        n_trout = 0,
        vision_smelt = 1,                   # vision
        vision_koaro = 1,
        vision_trout = 1,
        resource_pref_adult_smelt = 0.5,   # prey preference
        resource_pref_juv_smelt = 1.0,
        resource_pref_adult_koaro = 1.0,
        resource_pref_juv_koaro = 1.0,
        prey_pref_trout = 1.0,
        consume_amount_smelt = 1.0,        # resource consumption
        consume_amount_koaro = 1.0,
        Δenergy_smelt = 10.0,               # energy gained
        Δenergy_koaro = 10.0,
        Δenergy_trout = 10.0,
        Δenergy_loss_smelt = 10.0,          # energy lost              
        Δenergy_loss_koaro = 10.0,
        Δenergy_loss_trout = 1.0,          
        length_mean_smelt = 25.0,           # length
        length_sd_smelt = 0.0,
        length_mean_koaro = 25.0,
        length_sd_koaro = 0.0,
        length_mean_trout = 25.0,
        length_sd_trout = 0.0,
        growth_rate_smelt = 1.0,            # growth rate (mm/timestep)
        growth_rate_koaro = 1.0,
        growth_rate_trout = 1.0,
        size_maturity_smelt = 20.0,         # adult size
        size_maturity_koaro = 20.0,
        size_maturity_trout = 20.0,
        breed_prob_smelt = 0.003,            # breeding prob
        breed_prob_koaro = 0.003,
        breed_prob_trout = 0.003,
        fecundity_mean_smelt = 15.0,         # fecundity
        fecundity_sd_smelt  = 0.0,
        fecundity_mean_koaro = 10.0,
        fecundity_sd_koaro = 0.0,
        fecundity_mean_trout = 5.0,
        fecundity_sd_trout = 0.0,
        breed_mortality_smelt = 0.0,        # mortality spawning
        breed_mortality_koaro = 0.0,
        breed_mortality_trout = 0.0,
        mortality_random_smelt = 0.001,      # mortality random 
        mortality_random_koaro = 0.001,
        mortality_random_trout = 0.0,
        seed = trunc(Int, rand() * 10000)
    )
    res = run_and_save(initialised_model)
    res.exp .= 3
    res.rep .= i
    res = select!(res, Not([:count_smelt, :count_trout]))

    CSV.write(string("data_created\\exp3_koaro_alone", i, ".csv"), res)
end


# smelt alone ---------------------------------------------
for i in 1:10 

    initialised_model = initialize_model(
        dims = (50, 50),                    # landscape size
        max_littoral_depth = 30,            # basal resources  
        res_k_lit = 100.0,
        res_k_pel = 100.0,                
        res_grow_r_lit = 0.2,
        res_grow_r_pel = 0.2,
        n_smelt = 500,                      # initial fish abund
        n_koaro = 0,
        n_trout = 0,
        vision_smelt = 1,                   # vision
        vision_koaro = 1,
        vision_trout = 1,
        resource_pref_adult_smelt = 0.0,   # prey preference
        resource_pref_juv_smelt = 0.0,
        resource_pref_adult_koaro = 1.0,
        resource_pref_juv_koaro = 1.0,
        prey_pref_trout = 1.0,
        consume_amount_smelt = 1.0,        # resource consumption
        consume_amount_koaro = 1.0,
        Δenergy_smelt = 11.0,               # energy gained
        Δenergy_koaro = 10.0,
        Δenergy_trout = 10.0,
        Δenergy_loss_smelt = 10.0,          # energy lost              
        Δenergy_loss_koaro = 10.0,
        Δenergy_loss_trout = 1.0,          
        length_mean_smelt = 25.0,           # length
        length_sd_smelt = 0.0,
        length_mean_koaro = 25.0,
        length_sd_koaro = 0.0,
        length_mean_trout = 25.0,
        length_sd_trout = 0.0,
        growth_rate_smelt = 1.0,            # growth rate (mm/timestep)
        growth_rate_koaro = 1.0,
        growth_rate_trout = 1.0,
        size_maturity_smelt = 20.0,         # adult size
        size_maturity_koaro = 20.0,
        size_maturity_trout = 20.0,
        breed_prob_smelt = 0.003,            # breeding prob
        breed_prob_koaro = 0.003,
        breed_prob_trout = 0.003,
        fecundity_mean_smelt = 10.0,         # fecundity
        fecundity_sd_smelt  = 0.0,
        fecundity_mean_koaro = 10.0,
        fecundity_sd_koaro = 0.0,
        fecundity_mean_trout = 5.0,
        fecundity_sd_trout = 0.0,
        breed_mortality_smelt = 0.0,        # mortality spawning
        breed_mortality_koaro = 0.0,
        breed_mortality_trout = 0.0,
        mortality_random_smelt = 0.001,      # mortality random 
        mortality_random_koaro = 0.001,
        mortality_random_trout = 0.0,
        seed = trunc(Int, rand() * 10000)
    )

    res = run_and_save(initialised_model)
    res.exp .= 4
    res.rep .= i
    res = select!(res, Not([:count_koaro, :count_trout]))

    CSV.write(string("data_created\\exp4_smelt_alone", i, ".csv"), res)
end


# smelt and koaro ---------------------------------------------
for i in 1:10 

    initialised_model = initialize_model(
        dims = (50, 50),                    # landscape size
        max_littoral_depth = 30,            # basal resources  
        res_k_lit = 100.0,
        res_k_pel = 100.0,                
        res_grow_r_lit = 0.2,
        res_grow_r_pel = 0.2,
        n_smelt = 500,                      # initial fish abund
        n_koaro = 500,
        n_trout = 0,
        vision_smelt = 1,                   # vision
        vision_koaro = 1,
        vision_trout = 1,
        resource_pref_adult_smelt = 0.0,   # prey preference
        resource_pref_juv_smelt = 0.0,
        resource_pref_adult_koaro = 1.0,
        resource_pref_juv_koaro = 1.0,
        prey_pref_trout = 1.0,
        consume_amount_smelt = 1.0,        # resource consumption
        consume_amount_koaro = 1.0,
        Δenergy_smelt = 11.0,               # energy gained
        Δenergy_koaro = 10.0,
        Δenergy_trout = 10.0,
        Δenergy_loss_smelt = 10.0,          # energy lost              
        Δenergy_loss_koaro = 10.0,
        Δenergy_loss_trout = 1.0,          
        length_mean_smelt = 25.0,           # length
        length_sd_smelt = 0.0,
        length_mean_koaro = 25.0,
        length_sd_koaro = 0.0,
        length_mean_trout = 25.0,
        length_sd_trout = 0.0,
        growth_rate_smelt = 1.0,            # growth rate (mm/timestep)
        growth_rate_koaro = 1.0,
        growth_rate_trout = 1.0,
        size_maturity_smelt = 20.0,         # adult size
        size_maturity_koaro = 20.0,
        size_maturity_trout = 20.0,
        breed_prob_smelt = 0.003,            # breeding prob
        breed_prob_koaro = 0.003,
        breed_prob_trout = 0.003,
        fecundity_mean_smelt = 10.0,         # fecundity
        fecundity_sd_smelt  = 0.0,
        fecundity_mean_koaro = 10.0,
        fecundity_sd_koaro = 0.0,
        fecundity_mean_trout = 5.0,
        fecundity_sd_trout = 0.0,
        breed_mortality_smelt = 0.0,        # mortality spawning
        breed_mortality_koaro = 0.0,
        breed_mortality_trout = 0.0,
        mortality_random_smelt = 0.001,      # mortality random 
        mortality_random_koaro = 0.001,
        mortality_random_trout = 0.0,
        seed = trunc(Int, rand() * 10000)
    )

    res = run_and_save(initialised_model)
    res.exp .= 5
    res.rep .= i
    res = select!(res, Not([:count_trout]))

    CSV.write(string("data_created\\exp5_sment_and_koaro", i, ".csv"), res)
end



5+5