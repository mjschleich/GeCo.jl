# Things to consider:
# -- Different number of changed features --> continuous, categorical, different domain sizes
# -- What is the distance? How does it compare to the minimum distance?
# -- # of features changed, and is this equal to the expect number of changes
# -- number of generations required to find this explanation
# -- maybe: experiments with monotonicty  (for later)

using Pkg; Pkg.activate(".")
using GeCo, DataFrames, JLD

macro ClassifierGenerator(features, thresholds)
    return generateClassifierFunction(features, thresholds)
end

function generateClassifierFunction(features, thresholds)
    quote
        _instances ->
        begin
            ranges = Dict([(:AgeGroup, 3), (:EducationLevel, 3), (:MaxBillAmountOverLast6Months, 50810.0),
                        (:MaxPaymentAmountOverLast6Months, 51430.0), (:MonthsWithZeroBalanceOverLast6Months, 6.0),
                        (:MonthsWithLowSpendingOverLast6Months, 6.0), (:MonthsWithHighSpendingOverLast6Months, 6.0),
                        (:MostRecentBillAmount, 29450.0), (:MostRecentPaymentAmount, 15420.0),
                        (:TotalOverdueCounts, 3.0), (:TotalMonthsOverdue, 36.0),
                        (:HasHistoryOfOverduePayments, 1)])

            score = Array{Float64,1}(undef, nrow(_instances))
            for i in 1:nrow(_instances)
                condition_fails = false
                distance_sum = 0.0
                for (feat, thresh) in zip($(features), ($thresholds))
                    if _instances[i,feat] < thresh
                        condition_fails = true
                        distance_sum += abs(thresh - _instances[i,feat]) / ranges[feat]
                    end
                end
                if condition_fails
                    score[i] = max(0, 0.5 - (distance_sum / length($(features))))
                else
                    score[i] = 1
                end
            end
            score
        end
    end
end


function groundTruthExperiment(X, p, classifier, symbols, thresholds; norm_ratio=[0.5, 0.5, 0, 0])

    ranges = Dict(feature => Float64(maximum(col) - minimum(col)) for (feature, col) in pairs(eachcol(X)))
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

        if predictions[i] == 1 || any(X[i,sym] >= thresh for (sym, thresh) in zip(symbols, thresholds))
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
        for (sym, thresh) in zip(symbols, thresholds)
            # if direction[j] == 1
            if orig_instance[sym] <= thresh
                changed_needed += 1
                optimal_cf[sym] = thresh
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
        Number of correct outcomes:                 $(sum(correct_outcomes) / length(correct_outcomes))
        Average number of features changed:         $(mean(num_changed_used))
        Average number of features need to changed: $(mean(num_changed_needed))
        Average distances to original:              $(mean(distances_to_original))
        Average distances to optimal:               $(mean(distances_to_optimal))
        Average generations:                        $(mean(num_generation))
        ")
end

include("../credit/credit_setup_MACE.jl");

classifier_ordinal =  @ClassifierGenerator([:AgeGroup], [3])
classifier_numerical = @ClassifierGenerator([:MaxBillAmountOverLast6Months], [1500])
classifier_combined = @ClassifierGenerator([:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue], [1500, 3, 1500, 5])

groundTruthExperiment(X, p, classifier_ordinal, [:AgeGroup], [3])
groundTruthExperiment(X, p, classifier_numerical, [:MaxBillAmountOverLast6Months], [3000])
groundTruthExperiment(X, p, classifier_combined,
    [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue],
    [1500, 3, 1500, 5];
    norm_ratio=[0.2,0.8,0,0])




# function classifier_ordinal(instances::DataFrame)
#     ranges = Dict([(:AgeGroup, 3), (:EducationLevel, 3), (:MaxBillAmountOverLast6Months, 50810.0),
#                     (:MaxPaymentAmountOverLast6Months, 51430.0), (:MonthsWithZeroBalanceOverLast6Months, 6.0),
#                     (:MonthsWithLowSpendingOverLast6Months, 6.0), (:MonthsWithHighSpendingOverLast6Months, 6.0),
#                     (:MostRecentBillAmount, 29450.0), (:MostRecentPaymentAmount, 15420.0),
#                     (:TotalOverdueCounts, 3.0), (:TotalMonthsOverdue, 36.0),
#                     (:HasHistoryOfOverduePayments, 1)])
#     score = Array{Float64,1}(undef, nrow(instances))
#     for i in 1:nrow(instances)
#         if instances[i,:AgeGroup] >= 3
#             score[i] =  1
#         else
#             norm_distance = abs(3 - instances[i,:AgeGroup]) / ranges[:AgeGroup]
#             score[i] =  0.5 - 0.5 * norm_distance
#         end
#     end
#     return score
# end

# function classifier_numerical(instances::DataFrame)
#     ranges = Dict([(:AgeGroup, 3), (:EducationLevel, 3), (:MaxBillAmountOverLast6Months, 50810.0),
#                     (:MaxPaymentAmountOverLast6Months, 51430.0), (:MonthsWithZeroBalanceOverLast6Months, 6.0),
#                     (:MonthsWithLowSpendingOverLast6Months, 6.0), (:MonthsWithHighSpendingOverLast6Months, 6.0),
#                     (:MostRecentBillAmount, 29450.0), (:MostRecentPaymentAmount, 15420.0),
#                     (:TotalOverdueCounts, 3.0), (:TotalMonthsOverdue, 36.0),
#                     (:HasHistoryOfOverduePayments, 1)])
#     score = Array{Float64,1}(undef, nrow(instances))
#     for i in 1:nrow(instances)
#         if instances[i,:MaxBillAmountOverLast6Months] >= 3000
#             score[i] =  1
#         else
#             norm_distance = abs(3000 - instances[i,:MaxBillAmountOverLast6Months]) / ranges[:MaxBillAmountOverLast6Months]
#             score[i] =  0.5 - 0.5 * norm_distance
#         end
#     end
#     return score
# end

# function classifier_combined(instances::DataFrame)
#     ranges = Dict([(:AgeGroup, 3), (:EducationLevel, 3), (:MaxBillAmountOverLast6Months, 50810.0),
#                     (:MaxPaymentAmountOverLast6Months, 51430.0), (:MonthsWithZeroBalanceOverLast6Months, 6.0),
#                     (:MonthsWithLowSpendingOverLast6Months, 6.0), (:MonthsWithHighSpendingOverLast6Months, 6.0),
#                     (:MostRecentBillAmount, 29450.0), (:MostRecentPaymentAmount, 15420.0),
#                     (:TotalOverdueCounts, 3.0), (:TotalMonthsOverdue, 36.0),
#                     (:HasHistoryOfOverduePayments, 1)])

#     score = Array{Float64,1}(undef, nrow(instances))
#     for i in 1:nrow(instances)
#         if instances[i,:MaxBillAmountOverLast6Months] >= 1500 && instances[i,:AgeGroup] >= 3 && instances[i,:MostRecentBillAmount] >= 1500 && instances[i,:TotalMonthsOverdue] >= 5
#             score[i] =  1
#         else
#             norm_distance = mean(
#                 [max(0, 1500 - instances[i,:MaxBillAmountOverLast6Months]) / ranges[:MaxBillAmountOverLast6Months],
#                 max(0, 3 - instances[i,:AgeGroup]) / ranges[:AgeGroup],
#                 max(0, 1500 - instances[i,:MostRecentBillAmount]) / ranges[:MostRecentBillAmount],
#                 max(0, 5 - instances[i,:TotalMonthsOverdue]) / ranges[:TotalMonthsOverdue]])

#             score[i] =  max(0, 0.5 - norm_distance)
#         end
#     end
#     return score
# end