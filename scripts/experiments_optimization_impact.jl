
using Pkg; Pkg.activate(".")
using GeCo
using PyCall
using Printf
import Dates, JLD

function runBreakdownExperiment(dataset, desired_class)

    include("$(dataset)/$(dataset)_setup_PRF.jl")

    # features, groups = initializeFeatures(path*"/data_info.json", X)
    domains = initDomains(path, X)

    distance_temp = Array{Float64,1}(undef, 12)

    predictions = if mlj_classifier <: PyCall.PyObject
        mode.(ScikitLearn.predict(mlj_classifier, MLJ.matrix(X)))
    else
        MLJ.predict_mode(mlj_classifier, X)
    end

    first_neg = findfirst(predictions .!= desired_class)
    # predictions = broadcast(MLJ.mode, MLJ.predict(mlj_classifier,X), desired_class)
    # println("Total number of predictions: $(length(predictions)) \n",
    #     "Total number of positive predictions $(sum(predictions))")

    num_changed = Array{Int64,1}()
    feat_changed = Array{BitArray{1},1}()
    distances = Array{Float64,1}()
    correct_outcome = Array{Bool,1}()
    times = Array{Float64,1}()
    num_explored = Array{Int64,1}()
    num_generation = Array{Int64,1}()
    avg_rep_size = Array{Float64,1}()

    changed_feats = falses(size(X,2))

    suffix = "mlj_model"
    for clf in [mlj_classifier, classifier]

        println("Classifier: $suffix ($(Dates.hour(Dates.now())):$(Dates.minute(Dates.now())))")

        for compress in [false, true]

            num_explained = 0
            num_to_explain = 1000

            empty!(num_changed)
            empty!(feat_changed)
            empty!(distances)
            empty!(correct_outcome)
            empty!(times)
            empty!(num_explored)
            empty!(num_generation)
            empty!(avg_rep_size)

            println("compress_data: $compress ($(Dates.hour(Dates.now())):$(Dates.minute(Dates.now())))")

            # Run explanation once for compilation
            explain(orig_entity, X, path, clf;
                desired_class=desired_class,
                compress_data=compress,
                min_num_generations=5,
                max_num_generations=5,
                convergence_k=3,
                domains=domains)

            for i in 1:length(predictions)
                if predictions[i] != desired_class

                    (i % 100 == 0) && println("$(@sprintf("%.2f", 100*num_explained/num_to_explain))% through .. ")

                    orig_entity = X[i, :]

                    time = @elapsed (explanation, count, generation, rep_size) =
                        explain(orig_entity, X, path, clf;
                            desired_class=desired_class,
                            compress_data=compress,
                            max_num_generations=5,
                            min_num_generations=5,
                            convergence_k=3,
                            domains=domains)

                    # dist = distance(explanation[1:3, :], orig_entity, features, distance_temp; norm_ratio=[])
                    for (fidx, feat) in enumerate(propertynames(X))
                        changed_feats[fidx] = (orig_entity[feat] != explanation[1,feat])
                    end

                    ## We only consider the top-explanation for this
                    push!(correct_outcome, explanation[1,:outc])
                    push!(num_changed, sum(changed_feats))
                    push!(feat_changed, changed_feats)
                    push!(distances, explanation[1,:score])
                    push!(times, time)
                    push!(num_generation, generation)
                    push!(num_explored, count)
                    push!(avg_rep_size, mean(rep_size[1:generation+1]))

                    num_explained += 1
                    (num_explained >= num_to_explain) && break
                end
            end

            file = "scripts/results/breakdown_exp/geco_breakdown_experiments_$(suffix)_compress_$(compress).jld"
            JLD.save(file, "times", times, "dist", distances, "numfeat", num_changed, "num_generation", num_generation, "num_explored", num_explored,  "avg_rep_size", avg_rep_size)

            println("
                Average number of features changed: $(mean(num_changed))
                Average distances:                  $(mean(distances)) (normalized: $((mean(distances ./ size(X,2)))))
                Average times:                      $(mean(times))
                Average generations:                $(mean(num_generation))
                Average generated cfs:              $(mean(num_explored))
                Average representatino size:        $(mean(avg_rep_size))
                Correct outcomes:                   $(mean(correct_outcome))
                Saved to: $file")
        end

        suffix = "partial_model"
    end
end

# runBreakdownExperiment("allstate", 1)
runBreakdownExperiment("allstate_mlp",1)
