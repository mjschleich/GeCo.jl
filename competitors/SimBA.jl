using Pkg; Pkg.activate(".")
using GeCo, DataFrames, JLD, StatsBase

include("../scripts/credit/credit_setup_MACE.jl");

# Compute Feasible Space
# Pick random Feature Group
# Sample k points from feasible space for this feature group (let’s use k = 5 for now)
# Compute score for updated instance, and pick the best one
# If outcome = desired outcome, then stop, else repeat from 1.
function simBA(orig_instance, X, p, classifier, k, desired_class)

    feasible_space = feasibleSpace(X, orig_instance, p)

    population = DataFrame(orig_instance)

    insertcols!(population,
            :score => zeros(Float64, 1),
            :outc => falses(1),
            :estcf => falses(1),
            :mod => BitVector[falses(14) for _ = 1:1]
            )

    # we use the score function here to check whether return the good
    # and loop until some good is found
    population.score = GeCo.score(classifier, population, desired_class)

    index_set = Set(1:length(feasible_space.groups))
    # println(population[1,:score])
    # println(desired_class)
    num_changes = 0
    while (!isempty(index_set)) && ((desired_class == 1) ? (population[1,:score] < 0.5) : (population[1,:score] > 0.5))

        # get the index of the feature group to change
        index = rand(index_set)
        pop!(index_set, index)

        # check we should get how many of them
        space = feasible_space.feasibleSpace[index]
        feature_names = feasible_space.groups[index].names

        num_rows = min(size(space, 1), k)
        if (num_rows == 0)
            continue
        end

        rows = StatsBase.sample(1:nrow(space), StatsBase.FrequencyWeights(space.count), num_rows; replace=false)
        repeat!(population, num_rows)

        for i in 1:num_rows
            for fname in feature_names
                population[i,fname] = space[rows[i], fname]
            end
        end

        # calculate the scores
        population.score = GeCo.score(classifier, population, desired_class)

        # sort by the scores
        sort!(population, :score; rev=true)
        delete!(population, (2:num_rows))


        println(feature_names)
        num_changes += 1
    end
    return population, num_changes, population[1,:score]
end

res, num_changes, score = simBA(orig_instance, X, p, classifier, 10, 1)