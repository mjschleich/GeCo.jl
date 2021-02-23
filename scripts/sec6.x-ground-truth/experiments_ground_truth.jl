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
                failed_conditions = 0
                distance_sum = 0.0
                for (feat, thresh) in zip($(features), ($thresholds))
                    if _instances[i,feat] < thresh
                        failed_conditions += 1
                        distance_sum += abs(thresh - _instances[i,feat]) / ranges[feat]
                    end
                end
                if failed_conditions > 0
                    score[i] = max(0,0.5 - (0.5*distance_sum + 0.5*failed_conditions) / length($(features)))
                else
                    score[i] = 1
                end
            end
            score
        end
    end
end

# we want the threshold to be a high count value in the middle of the range (30% to 70%)
function threshGenerator(X, symbols)
    threshs = Array{Float64,1}()
    
    for symbol in symbols
        thresh = 0
        freq = 0
        space = combine(groupby(X[!,[symbol]], [symbol]; sort = true), nrow => :count)
        for row_index in Int(floor(0.3*nrow(space))) : Int(ceil(0.7*nrow(space)))
            if (freq < space[row_index, :count])
                freq = space[row_index, :count]
                thresh = space[row_index, symbol]
            end
        end
        append!(threshs, thresh)
    end
    return threshs
end


function groundTruthExperiment(X, p, classifier, symbols; 
    norm_ratio=[0.5, 0.5, 0, 0], 
    min_num_generations = 3, 
    max_num_samples::Int64=5,
    max_samples_init::Int64=20)

    thresholds = threshGenerator(X, symbols)
    #println(thresholds)

    ranges = Dict(feature => Float64(maximum(col) - minimum(col)) for (feature, col) in pairs(eachcol(X)))
    num_features = ncol(X)

    explained = 0

    num_recovered = 0
    correct_outcomes = Array{Bool,1}()
    num_changed_needed = Array{Int64,1}()
    num_changed_used = Array{Int64,1}()
    distances_to_optimal = Array{Float64,1}()
    distances_to_original = Array{Float64,1}()
    num_generation = Array{Int64,1}()
    times = Array{Float64,1}()

    predictions = classifier(X)


    for i in 1:nrow(X)
        if (explained >= 10)
            break
        end

        # we want to want the instances that fails all thresh
        if predictions[i] == 1 || any(X[i,sym] >= thresh for (sym, thresh) in zip(symbols, thresholds))
            continue
        end

        orig_instance = X[i, :]
        # run geco on that      
        time = @elapsed (explanation, count, generation, rep_size) = explain(orig_instance, X, p, classifier;
            norm_ratio=norm_ratio, min_num_generations = min_num_generations, max_num_samples=max_num_samples, max_samples_init=max_samples_init)
        
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
            if orig_instance[sym] < thresh
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
        # if explanation[1,1:length(optimal_cf)] == optimal_cf
        #     num_recovered += 1
        # end

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
        push!(times, time)
    end
    # println(num_recovered)
    # file = "ground_truth_ageGroup.jld"
    # JLD.save(file, "distances_used", distances_used, "distances_need", distances_need, "num_generation", num_generation, "num_changed", num_changed)

    println("
        Number of correct outcomes:                 $(sum(correct_outcomes) / length(correct_outcomes))
        Average time used:                          $(mean(times))
        Average number of features changed:         $(mean(num_changed_used))
        Average number of features need to changed: $(mean(num_changed_needed))
        Average distances to original:              $(mean(distances_to_original))
        Average distances to optimal:               $(mean(distances_to_optimal))
        Average generations:                        $(mean(num_generation))
        ")
end


include("../credit/credit_setup_MACE.jl");
ordinal_th = threshGenerator(X, [:AgeGroup])
numerical_th = threshGenerator(X, [:MaxBillAmountOverLast6Months])
th_2 = threshGenerator(X, [:AgeGroup, :MaxBillAmountOverLast6Months])
th_3 = threshGenerator(X, [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount])
th_4 = threshGenerator(X, [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue])
th_5 = threshGenerator(X, [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue])
th_6 = threshGenerator(X, [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MostRecentPaymentAmount])
classifier_ordinal =  @ClassifierGenerator([:AgeGroup], ordinal_th)
classifier_numerical = @ClassifierGenerator([:MaxBillAmountOverLast6Months], numerical_th)
classifier_length2 = @ClassifierGenerator([:AgeGroup, :MaxBillAmountOverLast6Months], th_2)
classifier_length3 = @ClassifierGenerator([:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount], th_3)
classifier_length4 = @ClassifierGenerator([:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue], th_4)
classifier_length5 = @ClassifierGenerator([:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue], th_5)
classifier_length6 = @ClassifierGenerator([:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MostRecentPaymentAmount], th_6)

println("Only first norm experiments")
groundTruthExperiment(X, p, classifier_ordinal, [:AgeGroup])
groundTruthExperiment(X, p, classifier_numerical, [:MaxBillAmountOverLast6Months])
groundTruthExperiment(X, p, classifier_length2,
    [:MaxBillAmountOverLast6Months, :AgeGroup];
    norm_ratio=[0,1.0,0,0])
groundTruthExperiment(X, p, classifier_length3,
    [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount];
    norm_ratio=[0,1.0,0,0])
groundTruthExperiment(X, p, classifier_length4,
    [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
    norm_ratio=[0,1.0,0,0])
groundTruthExperiment(X, p, classifier_length5,
    [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
    norm_ratio=[0,1.0,0,0])
groundTruthExperiment(X, p, classifier_length6,
    [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MostRecentPaymentAmount];    
    norm_ratio=[0,1.0,0,0])

println()
println("Zero and First Norm experiments")
groundTruthExperiment(X, p, classifier_ordinal, [:AgeGroup])
groundTruthExperiment(X, p, classifier_numerical, [:MaxBillAmountOverLast6Months])
groundTruthExperiment(X, p, classifier_length2,
    [:MaxBillAmountOverLast6Months, :AgeGroup];
    norm_ratio=[0.5,0.5,0,0])
groundTruthExperiment(X, p, classifier_length3,
    [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount];
    norm_ratio=[0.5,0.5,0,0])
groundTruthExperiment(X, p, classifier_length4,
    [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
    norm_ratio=[0.5,0.5,0,0])
groundTruthExperiment(X, p, classifier_length5,
    [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
    norm_ratio=[0.5,0.5,0,0])
groundTruthExperiment(X, p, classifier_length6,
    [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MostRecentPaymentAmount];
    norm_ratio=[0.5,0.5,0,0])

println("first norm experiments with large sample")
groundTruthExperiment(X, p, classifier_ordinal, [:AgeGroup];
    norm_ratio=[0,1.0,0,0],  max_num_samples = 100, max_samples_init = 300)
groundTruthExperiment(X, p, classifier_numerical, [:MaxBillAmountOverLast6Months];
    norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 100)
groundTruthExperiment(X, p, classifier_length2,
    [:MaxBillAmountOverLast6Months, :AgeGroup];
    norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 300)
groundTruthExperiment(X, p, classifier_length3,
    [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount];
    norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 300)
groundTruthExperiment(X, p, classifier_length4,
    [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
    norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 300)
groundTruthExperiment(X, p, classifier_length5,
    [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
    norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 300)
groundTruthExperiment(X, p, classifier_length6,
    [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MostRecentPaymentAmount];    
    norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 300)


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