
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
                    score[i] = 0.5 - (0.5 * distance_sum + 0.5 * failed_conditions) / length($features)
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
        space = combine(groupby(X, symbol; sort=true), nrow => :count)

        low = max(2,Int(floor(0.3*nrow(space))))
        high = min(nrow(space),Int(ceil(0.7 * nrow(space))))

        for row_index in low:high
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
    min_num_generations=5,
    max_samples_mut::Int64=5,
    max_samples_init::Int64=20,
    suffix::String="")

    ranges = Dict(feature => Float64(maximum(col) - minimum(col)) for (feature, col) in pairs(eachcol(X)))
    num_features = ncol(X)

    explained = 0

    num_recovered = 0
    correct_outcomes = Array{Bool,1}()
    num_changed_needed = Array{Int64,1}()
    num_changed_used = Array{Int64,1}()
    distances_to_optimal = Array{Float64,1}()
    distances_to_original = Array{Float64,1}()
    distances_optimal_to_orig = Array{Float64,1}()
    num_generation = Array{Int64,1}()
    times = Array{Float64,1}()

    predictions = classifier(X)

    for i in 1:nrow(X)

        if explained >= 1000
            break
        end

        # We consider only instances that fail all conditions
        if predictions[i] == 1 || any(X[i,sym] >= thresh for (sym, thresh) in zip(symbols, thresholds))
            continue
        end

        orig_instance = X[i, :]

        time = @elapsed (explanation, count, generation, rep_size) = explain(orig_instance, X, p, classifier;
            norm_ratio=norm_ratio,
            min_num_generations=1,
            max_num_generations=25,
            max_num_samples=max_samples_mut,
            max_samples_init=max_samples_init,
            convergence_k=1,
            size_distance_temp=10000000)

        changed = 0
        for i in 1:num_features
            if (orig_instance[i] != explanation[1,i])
                changed += 1
            end
        end

        changed_needed = 0
        optimal_cf = deepcopy(orig_instance)
        for (sym, thresh) in zip(symbols, thresholds)
            if orig_instance[sym] < thresh
                optimal_cf[sym] = thresh
                changed_needed += 1
            end
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

        dist_optimal_to_orig = distance(optimal_cf, orig_instance, num_features, ranges;
            norm_ratio=[0,1.0,0,0])

        explained += 1
        push!(correct_outcomes, explanation[1, :outc])
        push!(distances_to_original, dist_to_original)
        push!(distances_to_optimal, dist_to_optimal)
        push!(distances_optimal_to_orig, dist_optimal_to_orig)
        push!(num_generation, generation)
        push!(num_changed_used, changed)
        push!(num_changed_needed, changed_needed)
        push!(times, time)
    end

    println(explained)

    ratio =
        if norm_ratio == [0.0,1.0,0.0,0.0]
        "l1_norm"
    elseif norm_ratio == [0.5,0.5,0.0,0.0]
        "l0_l1_norm"
    else
        norm_ratio
    end

    exp_name = string(length(symbols)) * "features"
    file = "scripts/results/ground_truth_exp/credit_ground_truth_exp_symbols_$(exp_name)_ratio_$(ratio)_samples_$(max_samples_init)_$(max_samples_mut)_$suffix.jld"

    JLD.save(file,
        "distances_to_original", distances_to_original,
        "distances_to_optimal", distances_to_optimal,
        "distances_optimal_to_orig", distances_optimal_to_orig,
        "num_generation", num_generation,
        "num_changed_used", num_changed_used,
        "num_changed_needed", num_changed_needed,
        "outcomes", correct_outcomes,
        "times", times)

    println("
        Symbols:                                    $(symbols)  ExpName: $(exp_name)
        Norm:                                       $(norm_ratio)  Samples: ($(max_samples_init),$(max_samples_mut))
        Number of correct outcomes:                 $(sum(correct_outcomes) / length(correct_outcomes))
        Average time used:                          $(mean(times))
        Average number of features changed:         $(mean(num_changed_used))
        Average number of features need to changed: $(mean(num_changed_needed))
        Average distances to original:              $(mean(distances_to_original))
        Average distances to optimal:               $(mean(distances_to_optimal))   Optimal to Orig: $(mean(distances_optimal_to_orig))
        Average generations:                        $(mean(num_generation))
        \n")
end


include("../credit/credit_setup_MACE.jl");

p = initPLAF()
@PLAF(p, :cf.isMale .== :x.isMale)
@PLAF(p, :cf.isMarried .== :x.isMarried)

vars = [
    :MaxBillAmountOverLast6Months,
    :MostRecentBillAmount,
    :MaxPaymentAmountOverLast6Months,
    :MostRecentPaymentAmount,
    :TotalMonthsOverdue,
    :MonthsWithZeroBalanceOverLast6Months,
    :MonthsWithLowSpendingOverLast6Months,
    :MonthsWithHighSpendingOverLast6Months,
    :AgeGroup,
    :EducationLevel,
    :TotalOverdueCounts,
    :HasHistoryOfOverduePayments
]

l1_norm = [0.0, 1.0, 0.0, 0.0 ]
l0_l1_norm = [0.5, 0.5, 0.0, 0.0 ]
norms = [l1_norm, l0_l1_norm]

samples1 = (samples_mut = 5, samples_init = 20)
samples2 = (samples_mut = 15, samples_init = 60)
samples3 = (samples_mut = 25, samples_init = 100)
#samples3 = (samples_mut = 100, samples_init = 300)
samples = [samples1, samples2, samples3]

thresholds = thresholdGenerator(X, vars)
# thresholds = Dict(:MostRecentBillAmount => 4020.0, :MaxBillAmountOverLast6Months => 4320.0, :AgeGroup => 2.0, :TotalMonthsOverdue => 12.0, :MaxPaymentAmountOverLast6Months => 3050.0, :MostRecentPaymentAmount => 1220.0)

########
# EXPERIMENT WITH DECREASING ORDER WRT DOMAIN SIZE
########

symbols = [Tuple(vars[1:i]) for i in 1:length(vars)]
norms = [l1_norm]

# for norm_ratio in norms, num_samples in samples, syms in symbols

#     threshs = [thresholds[s] for s in syms]
#     this_classifier = @ClassifierGenerator(syms, threshs)

#     groundTruthExperiment(X, p, this_classifier, syms, threshs;
#         norm_ratio=norm_ratio,
#         max_samples_init=num_samples.samples_init,
#         max_samples_mut=num_samples.samples_mut,
#         suffix="_decreasing_domain_size"
#         )
# end



########
# EXPERIMENT WITH INTERLEAVED ORDER WRT DOMAIN SIZE
########

interleaved_vars = [
    :MaxBillAmountOverLast6Months,
    :TotalOverdueCounts,
    :MostRecentBillAmount,
    :AgeGroup,
    :MaxPaymentAmountOverLast6Months,
    :HasHistoryOfOverduePayments,
    :MostRecentPaymentAmount,
    :TotalMonthsOverdue,
    :EducationLevel,
    :MonthsWithZeroBalanceOverLast6Months,
    :MonthsWithLowSpendingOverLast6Months,
    :MonthsWithHighSpendingOverLast6Months,
]

symbols = [Tuple(interleaved_vars[1:i]) for i in 1:length(interleaved_vars)]

for norm_ratio in norms, num_samples in samples, syms in symbols

    threshs = [thresholds[s] for s in syms]
    this_classifier = @ClassifierGenerator(syms, threshs)

    groundTruthExperiment(X, p, this_classifier, syms, threshs;
        norm_ratio=norm_ratio,
        max_samples_init=num_samples.samples_init,
        max_samples_mut=num_samples.samples_mut,
        suffix="interleaved_domain_size"
        )
end
