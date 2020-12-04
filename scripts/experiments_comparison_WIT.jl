

using Pkg; Pkg.activate(".")
using GeneticCounterfactual

using Printf
import Dates, JLD

function runExperiment(dataset::String, desired_class::Int64, feasibility_check::Bool)

    include("$(dataset)/$(dataset)_setup_MACE.jl")

    num_changed = Array{Int64,1}()
    feat_changed = Array{BitArray{1},1}()
    distances = Array{Float64,1}()
    correct_outcome = Array{Bool,1}()
    times = Array{Float64,1}()

    features, groups = initializeFeatures(path*"/data_info.json", X)

    X.pred = ScikitLearn.predict(classifier, MLJ.matrix(X))

    println("Total number of predictions: $(size(X,1)) \n"*
        "Total number of positive predictions $(sum(X[!,:pred])) \n"*
        "Total number of negative predictions $(size(X,1)-sum(X[:,:pred]))")

    num_explained = 0
    num_to_explain = 5000
    num_failed_explained = 0

    for i in 1:size(X,1)
        if X[i, :pred] != desired_class
            # (i % 100 == 0) && println("$(@sprintf("%.2f", 100*num_explained/num_to_explain))% through .. ")
            orig_entity = X[i, :]
            time = @elapsed (clostest_entity, distance_min) = findMinimumObservable(X, orig_entity, features; label_name=:pred, desired_class=desired_class, check_feasibility=feasibility_check)

            if isnothing(clostest_entity)
                # println("Entity $i cannot be explained.")
                num_failed_explained += 1
            else
                changed = []
                for feature in propertynames(orig_entity)
                    if feature != :pred
                        push!(changed, orig_entity[feature] != clostest_entity[feature])
                    end
                end
                push!(num_changed, count(changed))
                push!(feat_changed, changed)
                push!(distances, distance_min)
                push!(times, time)
            end

            num_explained += 1
            (num_explained >= num_to_explain) && break
        end
    end

    file = "scripts/results/wit_exp/$(dataset)_wit_mace_experiment_feasible_$(feasibility_check)_local.jld"
    JLD.save(file, "times", times, "dist", distances, "numfeat", num_changed, "feat_changed", feat_changed, "num_failed", num_failed_explained)

    println("
        Average number of features changed: $(mean(num_changed))
        Average distances:                  $(mean(distances))
        Average times:                      $(mean(times))
        Number of failed explanations:      $(num_failed_explained) ($(100*num_failed_explained/num_explained)%)
        Saved to: $file")
end

runExperiment("credit", 1, true)
runExperiment("credit", 1, false)
runExperiment("adult", 1, true)
runExperiment("adult", 1, false)
