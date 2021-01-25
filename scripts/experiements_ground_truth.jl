using Pkg; Pkg.activate(".")
using GeCo
include("../scripts/credit/credit_setup_MACE.jl");
include("./credit/credit_constraints_MACE_counter.jl");
function get_bad_counter(second_level)
    origs = DataFrame()
    counters = DataFrame()
    desired_class = 1.0
    # filter some good labled outcome
    predictions = ScikitLearn.predict(classifier, MLJ.matrix(X))
    count = 0
    num_generateds = 0
    for i in 1:length(predictions)
        if (count >= 10)
            break
        end
        if predictions[i] != desired_class
            continue
        end
        count += 1
        num_generateds += 1
        good_counterpart = X[i, :]

        # get the feasible space for this good entity
        feasible_space = feasibleSpace(X, good_counterpart, p)
        groups = feasible_space.groups
        num_features = feasible_space.num_features


        ## get the closest counterpart with one feature changed
        bad_1 = nothing
        min_distance = 0.0
        # println(groups)
        # loop over each group
        for (index, group) in enumerate(groups)
            # println(index)
            df = feasible_space.feasibleSpace[index]
            test_instance = DataFrame(good_counterpart)
            repeat!(test_instance, nrow(df))
            # if the feasible space is empty, continue to next group
            isempty(df) && continue;
            # println(df)
            # for each of the row in feature space
            for row in 1:nrow(df)
                for feature in group.features
                    test_instance[row,feature] = df[row, feature]
                end
                # check the outcome and distance of the new instance
                # if (ScikitLearn.predict(classifier, MLJ.matrix(test_instance)) != [desired_class])
                #     distances = distance(good_counterpart, test_instance[1,:], num_features, feasible_space.ranges)
                #     # println(distances)
                #     if (min_distance == 0 || distances < min_distance)
                #         # this is what we want
                #         bad_1 = DataFrame(test_instance)
                #         min_distance = distances
                #     end
                # end
            end
            insertcols!(test_instance,
                :score => ScikitLearn.predict(classifier, MLJ.matrix(test_instance))
                )
            filter!(row -> row[:score] != desired_class, test_instance)
            if (isempty(test_instance))
                continue
            end
            distances = distance(test_instance, good_counterpart, num_features, feasible_space.ranges)
            min_index = argmin(distances)
            if (min_distance == 0 || distances[min_index] < min_distance)
                # this is what we want
                bad_1 = DataFrame(test_instance[min_index, 1:14])
                min_distance = distances[min_index]
            end
        end

        if (bad_1 !== nothing)
            append!(counters, bad_1)
            push!(origs, good_counterpart)
        end
        # println(ScikitLearn.predict(classifier, MLJ.matrix(bad_1)))
        # println(min_distance)
        # println(bad_1)
        # println(good_counterpart)

        
        if (!second_level)
            continue
        end
        bad_2 = nothing
        min_distance = 0.0
        ## find the instance with two feature changed
        for (index_1, group_1) in enumerate(groups)
            df_1 = feasible_space.feasibleSpace[index_1]
            isempty(df_1) && continue;
            for (index_2, group_2) in enumerate(groups)
                df_2 = feasible_space.feasibleSpace[index_2]
                (index_2 <= index_1 || isempty(df_2)) && continue;

                test_instance = DataFrame(good_counterpart)
                repeat!(test_instance, nrow(df_1)*nrow(df_2))
                # for each of the row in feature space
                rownum = 1
                for row_1 in 1:nrow(df_1)
                    for row_2 in 1:nrow(df_2)
                        for feature in group_1.features
                            test_instance[rownum,feature] = df_1[row_1, feature]
                        end
                        for feature in group_2.features
                            test_instance[rownum,feature] = df_2[row_2, feature]
                        end
                        # # check the outcome and distance of the new instance
                        # if (ScikitLearn.predict(classifier, MLJ.matrix(test_instance)) != desired_class)
                        #     distances = distance(good_counterpart, test_instance[1,:], num_features, feasible_space.ranges)
                        #     # println(distances)
                        #     if (min_distance == 0 || distances < min_distance)
                        #         # this is what we want
                        #         bad_2 = DataFrame(test_instance)
                        #         min_distance = distances
                        #         break
                        #     end
                        # end
                        rownum += 1
                    end
                end
                # filter and get the largets
                insertcols!(test_instance,
                    :score => ScikitLearn.predict(classifier, MLJ.matrix(test_instance))
                    )
                filter!(row -> row[:score] != desired_class, test_instance)
                if (isempty(test_instance))
                    continue
                end
                distances = distance(test_instance, good_counterpart, num_features, feasible_space.ranges)
                min_index = argmin(distances)
                if (min_distance == 0 || distances[min_index] < min_distance)
                    # this is what we want
                    bad_2 = DataFrame(test_instance[min_index, 1:14])
                    min_distance = distances[min_index]
                end
            end
        end
        # println(ScikitLearn.predict(classifier, MLJ.matrix(bad_1)))
        # println(min_distance)
        # println(bad_2)
        # println(good_counterpart)


        ## we get two counter-instance of the original one, run GeCo to see whether we can generate
        ## the counterfactuals as the original one
        if (bad_2 !== nothing)
            push!(origs, good_counterpart)
            append!(counters, bad_2)
        end
    end
    return (origs, counters)
end

function analysis_ground_truth(origs::DataFrame, counters::DataFrame)
    num_found = 0
    for i in 1:nrow(counters)
        explanation,  = explain(counters[i,:], X, p, classifier)
        # check whether the original exists
        for j in 1:nrow(explanation)
            if (explanation[j,1:14] == origs[i,:])
                num_found += 1
                break
            end
        end
    end
    return num_found
end

(origs, counters) = @time get_bad_counter(true)
include("../scripts/credit/credit_setup_MACE.jl");

num_found = @time analysis_ground_truth(origs, counters)
println(num_found)
println(size(counters)[1])
print(1.0*num_found/size(counters)[1])
