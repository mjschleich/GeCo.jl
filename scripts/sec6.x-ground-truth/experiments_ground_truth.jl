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
    clf = quote
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
                for (feat, thresh) in zip($features, ($thresholds))
                    if _instances[i,feat] < thresh
                        failed_conditions += 1
                        distance_sum += abs(thresh - _instances[i,feat]) / ranges[feat]
                    end
                end
                if failed_conditions > 0
                    score[i] = max(0,0.5 - (0.5*distance_sum + 0.5*failed_conditions) / length($features))
                else
                    score[i] = 1
                end
            end
            score
        end
    end
    # @show clf
    esc(clf)
end

# we want the threshold to be a high count value in the middle of the range (30% to 70%)
function thresholdGenerator(X, symbols)
    threshs = Dict{Symbol,Float64,}()

    for symbol in symbols
        thresh = 0
        freq = 0
        space = combine(groupby(X, symbol; sort = true), nrow => :count)
        for row_index in Int(floor(0.3*nrow(space))) : Int(ceil(0.7*nrow(space)))
            if (freq < space[row_index, :count])
                freq = space[row_index, :count]
                thresh = space[row_index, symbol]
            end
        end
        threshs[symbol] = thresh
    end

    println("symbols: $symbols thresholds: $threshs ")
    return threshs
end


function groundTruthExperiment(X, p, classifier, symbols, thresholds;
    norm_ratio=[0.5, 0.5, 0, 0],
    min_num_generations = 3,
    max_samples_mut::Int64=5,
    max_samples_init::Int64=20)

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
        if (explained >= 1000)
            break
        end

        # we want to want the instances that fails all thresh
        if predictions[i] == 1 || any(X[i,sym] >= thresh for (sym, thresh) in zip(symbols, thresholds))
            continue
        end

        orig_instance = X[i, :]
        # run geco on that
        time = @elapsed (explanation, count, generation, rep_size) = explain(orig_instance, X, p, classifier;
            norm_ratio=norm_ratio, min_num_generations = min_num_generations, max_num_samples=max_samples_mut, max_samples_init=max_samples_init)

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
        if explanation[1,1:length(optimal_cf)] == optimal_cf
            num_recovered += 1
        end

        # println("Symbol: $symbols\n exp: $(explanation[1,symbols])\n orig: $(orig_instance[symbols])\n")
        # println("Correct Outcome: $(explanation[1, :outc]) \n\n")

        dist_to_original = distance(explanation[1, :], orig_instance, num_features, ranges;
            norm_ratio=[0,1.0,0,0])

        dist_to_optimal = distance(explanation[1, :], optimal_cf, num_features, ranges;
            norm_ratio=[0,1.0,0,0])

        explained += 1
        push!(correct_outcomes, explanation[1, :outc])
        push!(distances_to_original, dist_to_original)
        push!(distances_to_optimal, dist_to_optimal)
        push!(num_generation, generation)
        push!(num_changed_used, changed)
        push!(num_changed_needed, changed_needed)
        push!(times, time)
    end

    ratio =
        if norm_ratio == [0.0,1.0,0.0,0.0]
            "l1_norm"
        elseif norm_ratio == [0.5,0.5,0.0,0.0]
            "l0_l1_norm"
        else
            norm_ratio
        end

    exp_name = if length(symbols) == 1
            symbols[1]
        else
            string(length(symbols))*"features"
        end

    file = "scripts/results/ground_truth_exp/credit_ground_truth_experiment_symbols_$(exp_name)_ratio_$(ratio)_samples_$(max_samples_init)_$(max_samples_mut).jld"
    JLD.save(file,
        "distances_to_original", distances_to_original,
        "distances_to_optimal", distances_to_optimal,
        "num_generation", num_generation,
        "num_changed_used", num_changed_used,
        "num_changed_needed", num_changed_needed,
        "times", times)

    println("
        Symbols:                                    $(symbols)  ExpName: $(exp_name)
        Norm:                                       $(norm_ratio)
        Number of correct outcomes:                 $(sum(correct_outcomes) / length(correct_outcomes))
        Average time used:                          $(mean(times))
        Average number of features changed:         $(mean(num_changed_used))
        Average number of features need to changed: $(mean(num_changed_needed))
        Average distances to original:              $(mean(distances_to_original))
        Average distances to optimal:               $(mean(distances_to_optimal))
        Average generations:                        $(mean(num_generation))
        \n")
end


include("../credit/credit_setup_MACE.jl");

# thresholds = thresholdGenerator(X, [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MostRecentPaymentAmount])
thresholds = Dict(:MostRecentBillAmount => 4020.0,:MaxBillAmountOverLast6Months => 4320.0,:AgeGroup => 2.0,:TotalMonthsOverdue => 12.0,:MaxPaymentAmountOverLast6Months => 3050.0,:MostRecentPaymentAmount => 1220.0)

syms1 = (:AgeGroup, )
syms2 = (:MaxBillAmountOverLast6Months, )
syms3 = (:MaxBillAmountOverLast6Months, :AgeGroup)
syms4 = (:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount)
syms5 = (:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue)
syms6 = (:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MaxPaymentAmountOverLast6Months)
sysm7 = (:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MaxPaymentAmountOverLast6Months, :MostRecentPaymentAmount)

l1_norm = [0.0, 1.0, 0.0, 0.0 ]
l0_l1_norm = [0.5, 0.5, 0.0, 0.0 ]

samples1 = (samples_mut = 5, samples_init = 20)
samples2 = (samples_mut = 10, samples_init = 40)
samples3 = (samples_mut = 100, samples_init = 300)

for norm_ratio = [l1_norm], num_samples=[samples1, samples2, samples3], syms in [syms1,syms2,syms3,syms4,syms5,syms6,syms7] # [syms1, syms2, syms3, syms4] [l1_norm, l0_l1_norm]

    # Comp thresholds:
    threshs = [thresholds[s] for s in syms]
    this_classifier = @ClassifierGenerator(syms, threshs)

    groundTruthExperiment(X, p, this_classifier, syms, threshs;
        norm_ratio=norm_ratio,
        max_samples_init=num_samples.samples_init,
        max_samples_mut=num_samples.samples_mut
        )
end

# classifier_ordinal =  @ClassifierGenerator([:AgeGroup], [thresholds[:AgeGroup]])
# classifier_numerical = @ClassifierGenerator([:MaxBillAmountOverLast6Months], [thresholds[:MaxBillAmountOverLast6Months]])
# classifier_length2 = @ClassifierGenerator([:AgeGroup, :MaxBillAmountOverLast6Months], th_2)
# classifier_length3 = @ClassifierGenerator([:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount], th_3)
# classifier_length4 = @ClassifierGenerator([:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue], th_4)
# classifier_length5 = @ClassifierGenerator([:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue], th_5)
# classifier_length6 = @ClassifierGenerator([:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MostRecentPaymentAmount], th_6)

# println("Only first norm experiments")
# groundTruthExperiment(X, p, classifier_ordinal, [:AgeGroup])
# groundTruthExperiment(X, p, classifier_numerical, [:MaxBillAmountOverLast6Months])
# groundTruthExperiment(X, p, classifier_length2,
#     [:MaxBillAmountOverLast6Months, :AgeGroup];
#     norm_ratio=[0,1.0,0,0])
# groundTruthExperiment(X, p, classifier_length3,
#     [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount];
#     norm_ratio=[0,1.0,0,0])
# groundTruthExperiment(X, p, classifier_length4,
#     [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
#     norm_ratio=[0,1.0,0,0])
# groundTruthExperiment(X, p, classifier_length5,
#     [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
#     norm_ratio=[0,1.0,0,0])
# groundTruthExperiment(X, p, classifier_length6,
#     [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MostRecentPaymentAmount];
#     norm_ratio=[0,1.0,0,0])

# println()
# println("Zero and First Norm experiments")
# groundTruthExperiment(X, p, classifier_ordinal, [:AgeGroup])
# groundTruthExperiment(X, p, classifier_numerical, [:MaxBillAmountOverLast6Months])
# groundTruthExperiment(X, p, classifier_length2,
#     [:MaxBillAmountOverLast6Months, :AgeGroup];
#     norm_ratio=[0.5,0.5,0,0])
# groundTruthExperiment(X, p, classifier_length3,
#     [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount];
#     norm_ratio=[0.5,0.5,0,0])
# groundTruthExperiment(X, p, classifier_length4,
#     [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
#     norm_ratio=[0.5,0.5,0,0])
# groundTruthExperiment(X, p, classifier_length5,
#     [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
#     norm_ratio=[0.5,0.5,0,0])
# groundTruthExperiment(X, p, classifier_length6,
#     [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MostRecentPaymentAmount];
#     norm_ratio=[0.5,0.5,0,0])

# println("first norm experiments with large sample")
# groundTruthExperiment(X, p, classifier_ordinal, [:AgeGroup];
#     norm_ratio=[0,1.0,0,0],  max_num_samples = 100, max_samples_init = 300)
# groundTruthExperiment(X, p, classifier_numerical, [:MaxBillAmountOverLast6Months];
#     norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 100)
# groundTruthExperiment(X, p, classifier_length2,
#     [:MaxBillAmountOverLast6Months, :AgeGroup];
#     norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 300)
# groundTruthExperiment(X, p, classifier_length3,
#     [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount];
#     norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 300)
# groundTruthExperiment(X, p, classifier_length4,
#     [:MaxBillAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
#     norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 300)
# groundTruthExperiment(X, p, classifier_length5,
#     [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue];
#     norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 300)
# groundTruthExperiment(X, p, classifier_length6,
#     [:MaxBillAmountOverLast6Months, :MaxPaymentAmountOverLast6Months, :AgeGroup, :MostRecentBillAmount, :TotalMonthsOverdue, :MostRecentPaymentAmount];
#     norm_ratio=[0,1.0,0,0], max_num_samples = 100, max_samples_init = 300)


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
# en d