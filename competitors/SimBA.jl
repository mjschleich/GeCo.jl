using Pkg; Pkg.activate(".")
using GeCo, DataFrames, JLD, StatsBase
include("../scripts/credit/credit_setup_MACE.jl");
# Compute Feasible Space 
# Pick random Feature Group 
# Sample k points from feasible space for this feature group (let’s use k = 5 for now)
# Compute score for updated instance, and pick the best one 
# If outcome = desired outcome, then stop, else repeat from 1. 
function simBA(orig_instance, X, p, classifier, k, desired_class)
    population = DataFrame(orig_instance)
    insertcols!(population,
            :score=>zeros(Float64, 1),
            :outc=>falses(1),
            :estcf=>falses(1),
            :mod=>BitVector[falses(14) for _=1:1]
            )
    # we use the score function here to check whether return the good
    # and loop until some good is found
    population[:score] = GeCo.score(classifier, population, desired_class)
    # println(population[1,:score])
    # println(desired_class)
    while ((desired_class == 1) ?  (population[1,:score] < 0.5) : (population[1,:score] > 0.5))
        feasible_space = feasibleSpace(X, population[1,:], p)
        # get the index of the feature group to change
        index = rand(1:size(feasible_space.groups)[1])
        # check we should get how many of them 
        space = feasible_space.feasibleSpace[index]
        num_rows = min(size(space)[1], k)
        if (num_rows == 0)
            continue
        end
        rows = StatsBase.sample(1:nrow(space), StatsBase.FrequencyWeights(space.count), num_rows)
        repeat!(population, num_rows)

        for i in 1:num_rows
            for fname in feasible_space.groups[index].names
                population[i,fname] = space[rows[i], fname]
            end
        end
        # print(size(population))
        # calculate and get the scores
        population[:score] = GeCo.score(classifier, population, desired_class)
        sort!(population, [:score])
        delete!(population, (2:num_rows))
    end
    return population
end

res = simBA(orig_instance, X, p, classifier, 5, 1)