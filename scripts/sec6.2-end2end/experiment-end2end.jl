
using Pkg; Pkg.activate(".")
using GeCo, DataFrames, Random

include("../../competitors/SimBA.jl")

Random.seed!(19)

include("../adult/adult_setup_MACE.jl")

orig_instance = X[7,:]

(explanation, count, generation, rep_size) = explain(orig_instance, X, p, classifier; max_num_samples=10, max_samples_init=40)

actions(unique(explanation), orig_instance)
println(generation)

# Random.seed!(19)

closest_entity, correct_outcome = simBA(orig_instance, X, p, classifier, 5, 1)

println("SimBA actions: $(correct_outcome)")
actions(DataFrame(closest_entity), orig_instance)

Random.seed!(19)
include("../adult/adult_setup_DICE.jl")

orig_instance = X[7,:]

(explanation, count, generation, rep_size) = explain(orig_instance, X, p, classifier; max_num_samples=10, max_samples_init=40)

actions(unique(explanation), orig_instance)

Random.seed!(19)

# 0.397959 corresponds to 40 hours / week
@PLAF(p, cf.hours_per_week <= 0.397959)

# education_Assoc, education_Bachelors, education_Doctorate, education_HS_grad, education_Masters, education_Prof_school, education_School, education_Some_college
@PLAF(p, if cf.education_Bachelors == 1; cf.age >= x.age + 0.0411 end)      ## Age + 3 years
@PLAF(p, if cf.education_Masters == 1; cf.age >= x.age + 0.0548 end)        ## Age + 4 years
@PLAF(p, if cf.education_Prof_school == 1; cf.age >= x.age + 0.0959 end)    ## Age + 7 years
@PLAF(p, if cf.education_Doctorate == 1; cf.age >= x.age + 0.0959 end)      ## Age + 7 years

(alt_explanation, count, generation, rep_size) = explain(orig_instance, X, p, classifier; max_num_samples=10, max_samples_init=40)

actions(unique(alt_explanation), orig_instance)