
using Pkg; Pkg.activate(".")
using GeCo, DataFrames, Random

Random.seed!(19)

include("../adult/adult_setup_MACE.jl")

orig_instance = X[7,:]

(explanation, count, generation, rep_size) = explain(orig_instance, X, p, classifier; max_num_samples=10, max_samples_init=40)

println(actions(unique(explanation), orig_instance))
println(generation)

Random.seed!(19)
include("../adult/adult_setup_DICE.jl")

orig_instance = X[7,:]

(explanation, count, generation, rep_size) = explain(orig_instance, X, p, classifier; max_num_samples=10, max_samples_init=40)

println(actions(unique(explanation), orig_instance))

Random.seed!(19)

@PLAF(p, cf.hours_per_week <= 0.45)

(alt_explanation, count, generation, rep_size) = explain(orig_instance, X, p, classifier; max_num_samples=10, max_samples_init=40)

println(actions(unique(alt_explanation), orig_instance))