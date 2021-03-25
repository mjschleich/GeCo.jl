
using Pkg; Pkg.activate(".")
using GeCo, Printf, DataFrames
import Dates, JLD, PyCall

function runExperimentConstraintEffect(X::DataFrame, programs::Vector{PLAFProgram}, prog_description::Vector{String}, dataset_name::String; desired_class::Int64=1)

    ranges = Dict(feature => Float64(maximum(col) - minimum(col)) for (feature, col) in pairs(eachcol(X)))
    num_features = ncol(X)

    distance_temp = Array{Float64,1}(undef, 12)

    ## Use for the MACE comparison:
    predictions = ScikitLearn.predict(classifier, MLJ.matrix(X))

    println("Total number of predictions: $(length(predictions))\n" *
        "Total number of positive predictions $(sum(predictions))\n" *
        "Total number of negative predictions $(length(predictions) - sum(predictions))")

    num_changed = Array{Int64,1}()
    feat_changed = Array{BitArray{1},1}()
    distances = Array{Float64,1}()
    correct_outcome = Array{Bool,1}()
    times = Array{Float64,1}()
    num_explored = Array{Int64,1}()
    num_generation = Array{Int64,1}()
    avg_rep_size = Array{Float64,1}()

    # Run explanation once for compilation
    explain(X[1, :], X, p, classifier; desired_class=desired_class, verbose=false)

    for (pid, p) in enumerate(programs), ratio in ["l1"], compress in [false]
        nratio =
            if ratio == "l1"
            [0.0, 1.0, 0.0, 0.0]
        elseif ratio == "l0l1"
            [0.5, 0.5, 0.0, 0.0]
        elseif ratio == "combined"
            [0.25, 0.25, 0.25, 0.25]
        end

        num_explained = 0
        num_to_explain = 500

        empty!(num_changed)
        empty!(feat_changed)
        empty!(distances)
        empty!(correct_outcome)
        empty!(times)
        empty!(num_explored)
        empty!(num_generation)
        empty!(avg_rep_size)

        domains = initDomains(p, X)

        for i in 1:length(predictions)
            if predictions[i] != desired_class
                (i % 100 == 0) && println("$(@sprintf("%.2f", 100 * num_explained / num_to_explain))% through .. ")

                orig_instance = X[i, :]
                time = @elapsed (explanation, count, generation, rep_size) =
                    explain(orig_instance, X, p, classifier;
                        domains=domains,
                        desired_class=desired_class,
                        verbose=false,
			min_num_generations=5,
			max_num_generations=5,
                        norm_ratio=nratio,
                        compress_data=compress)

                dist = distance(explanation[1:3, :], orig_instance, num_features, ranges;
                    norm_ratio=[0, 1.0, 0, 0],
                    distance_temp=distance_temp)

                changed_feats = falses(size(X, 2))
                for (fidx, feat) in enumerate(propertynames(X))
                    changed_feats[fidx] = (orig_instance[feat] != explanation[1,feat])
                end

                ## We only consider the top-explanation for this
                push!(correct_outcome, explanation[1,:outc])
                push!(feat_changed, changed_feats)
                push!(num_changed, sum(changed_feats))
                push!(distances, dist[1])
                push!(times, time)
                push!(num_generation, generation)
                push!(num_explored, count)
                push!(avg_rep_size, mean(rep_size[1:generation + 1]))

                num_explained += 1
                (num_explained >= num_to_explain) && break
            end
        end

        file = "scripts/results/constraint_exp/$(dataset_name)_constraint_experiment_program_$(prog_description[pid])_ratio_$(ratio)_compress_$(compress).jld"
        JLD.save(file, "times", times, "dist", distances, "numfeat", num_changed, "num_generation", num_generation, "num_explored", num_explored, "avg_rep_size", avg_rep_size)

        println("
            Average number of features changed: $(mean(num_changed))
            Average distances:                  $(mean(distances)) (normalized: $((mean(distances ./ size(X, 2)))))
            Average times:                      $(mean(times))
            Average generations:                $(mean(num_generation))
            Average generated cfs:              $(mean(num_explored))
            Average representatino size:        $(mean(avg_rep_size))
            Correct outcomes:                   $(mean(correct_outcome))
            Saved to: $file")
    end
end

for dataset in ["adult", "credit"]          # ["adult", "credit", "yelp"]

    if dataset ∈ ["adult", "credit"]
        include("../$(dataset)/$(dataset)_setup_MACE.jl")
    elseif dataset == "yelp"
        include("../$(dataset)/$(dataset)_setup_PRF.jl")
    end

    include("../$(dataset)/$(dataset)_constraints_variants.jl")

    runExperimentConstraintEffect(X, constraint_variations, constraint_descriptions, dataset)
end