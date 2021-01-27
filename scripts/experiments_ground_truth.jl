
# Things to consider:
# -- Different number of changed features --> continuous, categorical, different domain sizes
# -- What is the distance? How does it compare to the minimum distance?
# -- # of features changed, and is this equal to the expect number of changes
# -- number of generations required to find this explanation
# -- maybe: experiments with monotonicty  (for later)

# For now, let's do this for the credit dataset
## for credit dataset
using Pkg; Pkg.activate(".")
using GeCo
using JLD
include("../scripts/credit/credit_setup_MACE.jl");

# first check for one ordinal
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
            norm_distance = abs(40 - instances[i,:AgeGroup]) / ranges[:AgeGroup]
            score[i] =  0.5 - 0.5 * norm_distance
        end
    end
    return score
end

# first check for one numerical
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

function experiment()
    ##  for age one 
    explained = 0
    ranges = Dict(feature => Float64(maximum(col)-minimum(col)) for (feature, col) in pairs(eachcol(X)))
    num_features = ncol(X)
    num_changed = Array{Int64,1}()
    distances_need = Array{Float64,1}()
    distances_used = Array{Float64,1}()
    num_generation = Array{Int64,1}()

    for i in 1:nrow(X)
        if (explained >= 10)
            break
        end

        if (X[i, :AgeGroup] >= 3)
            continue
        end

        explained += 1
        
        ori_instance = X[i, :]
        # run geco on that
        (explanation, count, generation, rep_size) = explain(ori_instance, X, p, classifier_ordinal)

        changed = 0
        for i in 1:num_features
            if (ori_instance[i] != explanation[1,i])
                changed += 1
            end
        end

        dis_used = distance(explanation[1, :], ori_instance, num_features, ranges;norm_ratio=[0, 1.0, 0, 0])
        expect = DataFrame(ori_instance)

        expect[1, :AgeGroup] = 3
        # println(explanation[1, :AgeGroup])
        # println(expect[1, :AgeGroup])
        dis_need = distance(expect[1, :], ori_instance, num_features, ranges;norm_ratio=[0, 1.0, 0, 0])
        push!(distances_used, dis_used)
        push!(distances_need, dis_need)
        push!(num_generation, generation)
        push!(num_changed, changed)
    end
    # file = "ground_truth_ageGroup.jld"

    # JLD.save(file, "distances_used", distances_used, "distances_need", distances_need, "num_generation", num_generation, "num_changed", num_changed)

    println("
        Average number of features changed: $(mean(num_changed))
        Average distances used:                  $(mean(distances_used)) (normalized: $((mean(distances_used ./ size(X,2)))))
        Average distances need:                      $(mean(distances_need)) (normalized: $((mean(distances_need ./ size(X,2)))))
        Average generations:                $(mean(num_generation))
        ")

    ## for MaxBillAmountOverLast6Months(numerical one)
    explained = 0
    ranges = Dict(feature => Float64(maximum(col)-minimum(col)) for (feature, col) in pairs(eachcol(X)))
    num_features = ncol(X)
    num_changed = Array{Int64,1}()
    distances_need = Array{Float64,1}()
    distances_used = Array{Float64,1}()
    num_generation = Array{Int64,1}()

    for i in 1:nrow(X)
        if (explained >= 10)
            break
        end

        if (X[i, :MaxBillAmountOverLast6Months] >= 3000)
            continue
        end

        explained += 1
        
        ori_instance = X[i, :]
        # run geco on that
        (explanation, count, generation, rep_size) = explain(ori_instance, X, p, classifier_numerical)

        changed = 0
        for i in 1:num_features
            if (ori_instance[i] != explanation[1,i])
                changed += 1
            end
        end

        dis_used = distance(explanation[1, :], ori_instance, num_features, ranges;norm_ratio=[0, 1.0, 0, 0])
        expect = DataFrame(ori_instance)

        expect[1, :MaxBillAmountOverLast6Months] = 3000
        # println(explanation[1, :AgeGroup])
        # println(expect[1, :AgeGroup])
        dis_need = distance(expect[1, :], ori_instance, num_features, ranges;norm_ratio=[0, 1.0, 0, 0])
        push!(distances_used, dis_used)
        push!(distances_need, dis_need)
        push!(num_generation, generation)
        push!(num_changed, changed)
    end
    # file = "ground_truth_ageGroup.jld"

    # JLD.save(file, "distances_used", distances_used, "distances_need", distances_need, "num_generation", num_generation, "num_changed", num_changed)

    println("
        Average number of features changed: $(mean(num_changed))
        Average distances used:                  $(mean(distances_used)) (normalized: $((mean(distances_used ./ size(X,2)))))
        Average distances need:                      $(mean(distances_need)) (normalized: $((mean(distances_need ./ size(X,2)))))
        Average generations:                $(mean(num_generation))
        ")
end

experiment()

