

function runExperimentMACE(X::DataFrame, p::PLAFProgram, desired_class::Int64, dataset_name::String)

    domains = initDomains(p,X)
    ranges = Dict(feature => Float64(maximum(col)-minimum(col)) for (feature, col) in pairs(eachcol(X)))
    num_features = ncol(X)

    distance_temp = Array{Float64,1}(undef, 12)

    ## Use for the MACE comparison:
    predictions = ScikitLearn.predict(classifier, MLJ.matrix(X))

    println("Total number of predictions: $(length(predictions))\n"*
        "Total number of positive predictions $(sum(predictions))\n"*
        "Total number of negative predictions $(length(predictions)-sum(predictions))")

    num_changed = Array{Int64,1}()
    feat_changed = Array{BitArray{1},1}()
    distances = Array{Float64,1}()
    correct_outcome = Array{Bool,1}()
    times = Array{Float64,1}()
    num_explored = Array{Int64,1}()
    num_generation = Array{Int64,1}()
    avg_rep_size = Array{Float64,1}()

    # Run explanation once for compilation
    explain(X[1, :], X, p, classifier; desired_class = desired_class, verbose=false)

    for ratio in ["l0l1", "l1", "combined"]
        nratio =
            if ratio == "l1"
                [0.0, 1.0, 0.0, 0.0]
            elseif ratio == "l0l1"
                [0.5, 0.5, 0.0, 0.0]
            elseif ratio == "combined"
                [0.25, 0.25, 0.25, 0.25]
            end

        for compress in [false]

            num_explained = 0
            num_to_explain = 5000

            empty!(num_changed)
            empty!(feat_changed)
            empty!(distances)
            empty!(correct_outcome)
            empty!(times)
            empty!(num_explored)
            empty!(num_generation)
            empty!(avg_rep_size)

            for i in 1:length(predictions)
                if predictions[i] != desired_class
                    (i % 100 == 0) && println("$(@sprintf("%.2f", 100*num_explained/num_to_explain))% through .. ")

                    orig_instance = X[i, :]
                    time = @elapsed (explanation, count, generation, rep_size) =
                        explain(orig_instance, X, p, classifier;
                            domains=domains,
                            desired_class=desired_class,
                            verbose=false,
                            norm_ratio=nratio,
                            compress_data=compress)

                    dist = distance(explanation[1:3, :], orig_instance, num_features, ranges;
                        norm_ratio=[0, 1.0, 0, 0],
                        distance_temp=distance_temp)

                    changed_feats = falses(size(X,2))
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
                    push!(avg_rep_size, mean(rep_size[1:generation+1]))

                    num_explained += 1
                    (num_explained >= num_to_explain) && break
                end
            end

            #file = "scripts/results/mace_exp/$(dataset)_geco_mace_experiment_$(Dates.today())_$(Dates.hour(Dates.now())):$(Dates.minute(Dates.now())).jld"
            file = "scripts/results/mace_exp/$(dataset_name)_geco_mace_experiment_ratio_$(ratio)_compress_$(compress).jld"

            JLD.save(file, "times", times, "dist", distances, "numfeat", num_changed, "num_generation", num_generation, "num_explored", num_explored, "avg_rep_size", avg_rep_size)

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
    end
end
