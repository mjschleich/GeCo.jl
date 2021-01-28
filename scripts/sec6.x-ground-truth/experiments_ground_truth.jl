
# Things to consider:
# -- Different number of changed features --> continuous, categorical, different domain sizes
# -- What is the distance? How does it compare to the minimum distance?
# -- # of features changed, and is this equal to the expect number of changes
# -- number of generations required to find this explanation
# -- maybe: experiments with monotonicty  (for later)

using Pkg; Pkg.activate(".")
using GeCo, DataFrames, JLD


function classifier_ordinal(instances::DataFrame)
    ranges = Dict([(:AgeGroup, 3), (:EducationLevel,3), (:MaxBillAmountOverLast6Months, 50810.0),
                    (:MaxPaymentAmountOverLast6Months, 51430.0), (:MonthsWithZeroBalanceOverLast6Months,6.0),
                    (:MonthsWithLowSpendingOverLast6Months, 6.0), (:MonthsWithHighSpendingOverLast6Months, 6.0),
                    (:MostRecentBillAmount, 29450.0), (:MostRecentPaymentAmount, 15420.0),
                    (:TotalOverdueCounts, 3.0), (:TotalMonthsOverdue, 36.0),
                    (:HasHistoryOfOverduePayments, 1)])
    score = Array{Float64, 1}(undef, nrow(instances))
    for i in 1:nrow(instances)
        if instances[i,:AgeGroup] >= 3
            score[i] =  1
        else
            norm_distance = abs(3 - instances[i,:AgeGroup]) / ranges[:AgeGroup]
            score[i] =  0.5 - 0.5 * norm_distance
        end
    end
    return score
end

function classifier_numerical(instances::DataFrame)
    ranges = Dict([(:AgeGroup, 3), (:EducationLevel,3), (:MaxBillAmountOverLast6Months, 50810.0),
                    (:MaxPaymentAmountOverLast6Months, 51430.0), (:MonthsWithZeroBalanceOverLast6Months,6.0),
                    (:MonthsWithLowSpendingOverLast6Months, 6.0), (:MonthsWithHighSpendingOverLast6Months, 6.0),
                    (:MostRecentBillAmount, 29450.0), (:MostRecentPaymentAmount, 15420.0),
                    (:TotalOverdueCounts, 3.0), (:TotalMonthsOverdue, 36.0),
                    (:HasHistoryOfOverduePayments, 1)])
    score = Array{Float64, 1}(undef, nrow(instances))
    for i in 1:nrow(instances)
        if instances[i,:MaxBillAmountOverLast6Months] >= 3000
            score[i] =  1
        else
            norm_distance = abs(3000 - instances[i,:MaxBillAmountOverLast6Months]) / ranges[:MaxBillAmountOverLast6Months]
            score[i] =  0.5 - 0.5 * norm_distance
        end
    end
    return score
end

function classifier_combined(instances::DataFrame)
    ranges = Dict([(:AgeGroup, 3), (:EducationLevel,3), (:MaxBillAmountOverLast6Months, 50810.0),
                    (:MaxPaymentAmountOverLast6Months, 51430.0), (:MonthsWithZeroBalanceOverLast6Months,6.0),
                    (:MonthsWithLowSpendingOverLast6Months, 6.0), (:MonthsWithHighSpendingOverLast6Months, 6.0),
                    (:MostRecentBillAmount, 29450.0), (:MostRecentPaymentAmount, 15420.0),
                    (:TotalOverdueCounts, 3.0), (:TotalMonthsOverdue, 36.0),
                    (:HasHistoryOfOverduePayments, 1)])

    score = Array{Float64, 1}(undef, nrow(instances))
    for i in 1:nrow(instances)
        if instances[i,:MaxBillAmountOverLast6Months] >= 1500 && instances[i,:AgeGroup] >= 3 && instances[i,:MostRecentBillAmount] >= 1500 && instances[i,:TotalMonthsOverdue] >= 5
            score[i] =  1
        else
            norm_distance = mean(
                [max(0,1500-instances[i,:MaxBillAmountOverLast6Months]) / ranges[:MaxBillAmountOverLast6Months],
                max(0,3-instances[i,:AgeGroup]) / ranges[:AgeGroup],
                max(0,1500-instances[i,:MostRecentBillAmount]) / ranges[:MostRecentBillAmount],
                max(0,5-instances[i,:TotalMonthsOverdue]) / ranges[:TotalMonthsOverdue]])

            score[i] =  max(0, 0.5 - norm_distance)
        end
    end
    return score
end


function groundTruthExperiment(X, p, classifier, symbols, thresholds, direction; norm_ratio=[0.5, 0.5, 0, 0])

    ranges = Dict(feature => Float64(maximum(col)-minimum(col)) for (feature, col) in pairs(eachcol(X)))
    num_features = ncol(X)

    explained = 0

    correct_outcomes = Array{Bool,1}()
    num_changed_needed = Array{Int64,1}()
    num_changed_used = Array{Int64,1}()
    distances_to_optimal = Array{Float64,1}()
    distances_to_original = Array{Float64,1}()
    num_generation = Array{Int64,1}()

    predictions = classifier(X)


    for i in 1:nrow(X)
        if (explained >= 1000)
            break
        end

        if predictions[i] == 1 || any(X[i,sym] >= thresholds[sym] for sym in symbols)
            continue
        end

        orig_instance = X[i, :]

        # run geco on that
        (explanation, count, generation, rep_size) = explain(orig_instance, X, p, classifier;
            norm_ratio=norm_ratio)

        changed = 0
        for i in 1:num_features
            if (orig_instance[i] != explanation[1,i])
                changed += 1
            end
        end

        changed_needed = 0
        optimal_cf = deepcopy(orig_instance)
        for sym in symbols
            # if direction[j] == 1
                if orig_instance[sym] <= thresholds[sym]
                    changed_needed += 1
                    optimal_cf[sym] = thresholds[sym]
                end
            # else
            #     if orig_instance[sym] >= thresholds[sym]
            #         changed_needed += 1
            #         optimal_cf[sym] = thresholds[sym]
            #     end
            # end

            # println("Symbol: $sym optimal: $(thresholds[j]) exp: $(explanation[1,sym]) orig: $(orig_instance[sym])")

        end

        # println("Symbol: $symbols\n exp: $(explanation[1,symbols])\n orig: $(orig_instance[symbols])\n")
        # println("Correct Outcome: $(explanation[1, :outc]) \n\n")

        dist_to_original = distance(explanation[1, :], orig_instance, num_features, ranges; norm_ratio=norm_ratio)
        dist_to_optimal = distance(explanation[1, :], optimal_cf, num_features, ranges; norm_ratio=[0,1.0,0,0])

        explained += 1
        push!(correct_outcomes, explanation[1, :outc])
        push!(distances_to_original, dist_to_original)
        push!(distances_to_optimal, dist_to_optimal)
        push!(num_generation, generation)
        push!(num_changed_used, changed)
        push!(num_changed_needed, changed_needed)
    end

    # file = "ground_truth_ageGroup.jld"
    # JLD.save(file, "distances_used", distances_used, "distances_need", distances_need, "num_generation", num_generation, "num_changed", num_changed)

    println("
        Number of correct outcomes:                 $(sum(correct_outcomes)/length(correct_outcomes))
        Average number of features changed:         $(mean(num_changed_used))
        Average number of features need to changed: $(mean(num_changed_needed))
        Average distances to original:              $(mean(distances_to_original))
        Average distances to optimal:               $(mean(distances_to_optimal))
        Average generations:                        $(mean(num_generation))
        ")
end

include("../credit/credit_setup_MACE.jl");

# groundTruthExperiment(X, p, classifier_ordinal, [:AgeGroup], [3], [1])
# groundTruthExperiment(X, p, classifier_numerical, [:MaxBillAmountOverLast6Months], [3000], [1])
groundTruthExperiment(X,
    PLAFProgram(),
    classifier_combined,
    [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue],
    Dict(:MaxBillAmountOverLast6Months => 1500, :AgeGroup => 3, :MostRecentBillAmount => 1500, :TotalMonthsOverdue => 5),
    [1, 1, 1, 1];
    norm_ratio=[0.2,0.8,0,0])
















# function experiment()
#     ##  for age one
#     explained = 0
#     ranges = Dict(feature => Float64(maximum(col)-minimum(col)) for (feature, col) in pairs(eachcol(X)))
#     num_features = ncol(X)
#     num_changed = Array{Int64,1}()
#     distances_need = Array{Float64,1}()
#     distances_used = Array{Float64,1}()
#     num_generation = Array{Int64,1}()

#     for i in 1:nrow(X)
#         if (explained >= 10)
#             break
#         end

#         if (X[i, :AgeGroup] >= 3)
#             continue
#         end

#         explained += 1

#         ori_instance = X[i, :]
#         # run geco on that
#         (explanation, count, generation, rep_size) = explain(ori_instance, X, p, classifier_ordinal)

#         changed = 0
#         for i in 1:num_features
#             if (ori_instance[i] != explanation[1,i])
#                 changed += 1
#             end
#         end

#         dis_used = distance(explanation[1, :], ori_instance, num_features, ranges;norm_ratio=[0, 1.0, 0, 0])
#         expect = DataFrame(ori_instance)

#         expect[1, :AgeGroup] = 3
#         # println(explanation[1, :AgeGroup])
#         # println(expect[1, :AgeGroup])
#         dis_need = distance(expect[1, :], ori_instance, num_features, ranges;norm_ratio=[0, 1.0, 0, 0])
#         push!(distances_used, dis_used)
#         push!(distances_need, dis_need)
#         push!(num_generation, generation)
#         push!(num_changed, changed)
#     end
#     # file = "ground_truth_ageGroup.jld"

#     # JLD.save(file, "distances_used", distances_used, "distances_need", distances_need, "num_generation", num_generation, "num_changed", num_changed)

#     println("
#         Average number of features changed: $(mean(num_changed))
#         Average distances used:                  $(mean(distances_used)) (normalized: $((mean(distances_used ./ size(X,2)))))
#         Average distances need:                      $(mean(distances_need)) (normalized: $((mean(distances_need ./ size(X,2)))))
#         Average generations:                $(mean(num_generation))
#         ")

#     ## for MaxBillAmountOverLast6Months(numerical one)
#     explained = 0
#     ranges = Dict(feature => Float64(maximum(col)-minimum(col)) for (feature, col) in pairs(eachcol(X)))
#     num_features = ncol(X)
#     num_changed = Array{Int64,1}()
#     distances_to_optimal = Array{Float64,1}()
#     distances_to_original = Array{Float64,1}()
#     num_generation = Array{Int64,1}()

#     for i in 1:nrow(X)
#         if (explained >= 10)
#             break
#         end
#         if (X[i, :MaxBillAmountOverLast6Months] >= 3000)
#             continue
#         end

#         explained += 1

#         ori_instance = X[i, :]
#         # run geco on that
#         (explanation, count, generation, rep_size) = explain(ori_instance, X, p, classifier_numerical)

#         changed = 0
#         for i in 1:num_features
#             if (ori_instance[i] != explanation[1,i])
#                 changed += 1
#             end
#         end

#         dis_used = distance(explanation[1, :], ori_instance, num_features, ranges;norm_ratio=[0, 1.0, 0, 0])
#         expect = DataFrame(ori_instance)

#         expect[1, :MaxBillAmountOverLast6Months] = 3000
#         # println(explanation[1, :AgeGroup])
#         # println(expect[1, :AgeGroup])
#         dis_need = distance(expect[1, :], ori_instance, num_features, ranges;norm_ratio=[0, 1.0, 0, 0])
#         push!(distances_to_original, dis_used)
#         push!(distances_to_optimal, dis_need)
#         push!(num_generation, generation)
#         push!(num_changed, changed)
#     end
#     # file = "ground_truth_ageGroup.jld"

#     # JLD.save(file, "distances_used", distances_used, "distances_need", distances_need, "num_generation", num_generation, "num_changed", num_changed)

#     println("
#         Average number of features changed: $(mean(num_changed))
#         Average distances used:             $(mean(distances_to_original)) (normalized: $((mean(distances_to_original ./ size(X,2)))))
#         Average distances need:             $(mean(distances_to_optimal)) (normalized: $((mean(distances_to_optimal ./ size(X,2)))))
#         Average generations:                $(mean(num_generation))
#         ")



#     ## for combined
#     explained = 0
#     ranges = Dict(feature => Float64(maximum(col)-minimum(col)) for (feature, col) in pairs(eachcol(X)))
#     num_features = ncol(X)

#     distances_need = Array{Float64,1}()
#     distances_used = Array{Float64,1}()
#     num_generation = Array{Int64,1}()


#     for i in 1:nrow(X)
#         if (explained >= 10)
#             break
#         end

#         if (classifier_combined(DataFrame(X[i, :])) == 1)
#             continue
#         end

#         explained += 1

#         ori_instance = X[i, :]
#         # run geco on that
#         (explanation, count, generation, rep_size) = explain(ori_instance, X, p, classifier_combined;norm_ratio=[0, 1.0, 0, 0])
#         println(classifier_combined(explanation))
#         changed = 0
#         for i in 1:num_features
#             if (ori_instance[i] != explanation[1,i])
#                 changed += 1
#             end
#         end

#         changed_needed = 0
#         ## TODO: this should be iterate over the clas list




#         dis_used = distance(explanation[1, :], ori_instance, num_features, ranges;norm_ratio=[0, 1.0, 0, 0])
#         expect = DataFrame(ori_instance)
#         if (ori_instance[:MaxBillAmountOverLast6Months] < 1500)
#             changed_needed += 1
#             expect[1, :MaxBillAmountOverLast6Months] = 1500
#         end
#         if (ori_instance[:AgeGroup] < 3)
#             changed_needed += 1
#             expect[1, :AgeGroup] = 3
#         end
#         if (ori_instance[:MostRecentBillAmount] < 1500)
#             changed_needed += 1
#             expect[1, :MostRecentBillAmount] = 1500
#         end
#         if (ori_instance[:TotalMonthsOverdue] > 10)
#             changed_needed += 1
#             expect[1, :TotalMonthsOverdue] = 10
#         end





#         # println(explanation[1, :AgeGroup])
#         # println(expect[1, :AgeGroup])
#         dis_need = distance(expect[1, :], ori_instance, num_features, ranges;norm_ratio=[0, 1.0, 0, 0])
#         push!(distances_used, dis_used)
#         push!(distances_need, dis_need)
#         push!(num_generation, generation)
#         push!(num_changed_used, changed)
#         push!(num_changed_needed, changed_needed)
#     end
#     # file = "ground_truth_ageGroup.jld"

#     # JLD.save(file, "distances_used", distances_used, "distances_need", distances_need, "num_generation", num_generation, "num_changed", num_changed)

#     println("
#         Average number of features changed: $(mean(num_changed_used))
#         Average number of features need to changed: $(mean(num_changed_needed))
#         Average distances used:                  $(mean(distances_used)) (normalized: $((mean(distances_used ./ size(X,2)))))
#         Average distances need:                      $(mean(distances_need)) (normalized: $((mean(distances_need ./ size(X,2)))))
#         Average generations:                $(mean(num_generation))
#         ")
# end