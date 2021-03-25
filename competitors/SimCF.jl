using GeCo, DataFrames, JLD, StatsBase

# Compute Feasible Space
# Pick random Feature Group
# Sample k points from feasible space for this feature group (let’s use k = 5 for now)
# Compute score for updated instance, and pick the best one
# If outcome = desired outcome, then stop, else repeat from 1.
function simCF(orig_instance, X, p, classifier, k, desired_class)

    feasible_space = feasibleSpace(X, orig_instance, p)

    population = DataFrame(orig_instance)

    insertcols!(population,
        :score => zeros(Float64, 1),
        :outc => falses(1),
        :estcf => falses(1),
        :mod => BitVector[falses(feasible_space.num_features)]
    )

    # we use the score function here to check whether return the good
    # and loop until some good is found
    population.score = GeCo.score(classifier, population, desired_class)
    population.outc = population.score .> 0.5

    # println(population.outc)

    distance_temp=Vector{Float64}(undef, 25)

    index_set = Set(1:length(feasible_space.groups))

    num_changes = 0
    while (!isempty(index_set)) && !population.outc[1]

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

        # println(index, feature_names, space[rows, :])

        dist = distance(population, orig_instance, feasible_space.num_features, feasible_space.ranges;
            distance_temp=distance_temp, norm_ratio=[0,1.0,0,0])

        preds = GeCo.score(classifier, population, desired_class)

        for i in 1:nrow(population)
            p = (preds[i] < 0.5)
            population.score[i] = dist[i] + p * (2.0 - preds[i])
            population.outc[i] = !p
        end

        # println(population.score)

        # sort by the scores
        sort!(population, :score)
        delete!(population, (2:num_rows))

        num_changes += 1
    end

    return population[1,:], population.outc[1]
end
