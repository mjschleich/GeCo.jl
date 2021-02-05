
using Pkg; Pkg.activate(".")
using GeCo,Printf, DataFrames
import Dates, JLD, PyCall

function runBreakdownExperiment(X::DataFrame, p::PLAFProgram, classifier, dataset_name::String, model::String, desired_class::Int64)

    if isfile("data/$dataset_name/domains.jld")
        println("Loading domains from file: data/$dataset_name/domains.jld ($(Dates.now()))")
        d = JLD.load("data/$dataset_name/domains.jld")
        domains = d["domains"]
    else
        println("Initializing Domains: ($(Dates.now()))")
        domains = initDomains(p, X)

        println("Saving Domains: ($(Dates.now()))")
        JLD.save("data/$dataset_name/domains.jld", "domains", domains)
    end

    println("Computing Predictions: ($(Dates.now()))")
    predictions = if classifier isa PyCall.PyObject
            mode.(ScikitLearn.predict(classifier, MLJ.matrix(X)))
        else
            MLJ.predict_mode(classifier, X)
        end

    first_neg = findfirst(predictions .!= desired_class)

    # println("Total number of predictions: $(length(predictions)) \n",
    #     "Total number of positive predictions $(sum(predictions))")

    num_to_explain = 1000

    num_changed = Array{Int64,1}(undef, num_to_explain)
    feat_changed = Array{BitArray{1},1}(undef, num_to_explain)
    distances = Array{Float64,1}(undef, num_to_explain)
    correct_outcome = Array{Bool,1}(undef, num_to_explain)
    times = Array{Float64,1}(undef, num_to_explain)
    num_explored = Array{Int64,1}(undef, num_to_explain)
    num_generation = Array{Int64,1}(undef, num_to_explain)
    avg_rep_size = Array{Float64,1}(undef, num_to_explain)

    prep_time = Array{Float64,1}(undef, num_to_explain)
    selection_time = Array{Float64,1}(undef, num_to_explain)
    mutation_time = Array{Float64,1}(undef, num_to_explain)
    crossover_time = Array{Float64,1}(undef, num_to_explain)

    for partial in [false, true], compress in [true, false], mutation_run in [true, false], crossover_run in [true, false]

        (!mutation_run && !crossover_run) && continue

        println("partial: $partial mutation_run: $mutation_run crossover_run $crossover_run compress_data: $compress ($(Dates.now()))")

        num_explained = 0
        changed_feats = falses(size(X,2))

        orig_instance = X[first_neg, :]
        clf =
            if partial
                if classifier isa PyCall.PyObject
                    initMLPEval(classifier, orig_instance)
                else
                    initPartialRandomForestEval(classifier, orig_instance, 1)
                end
            else
                classifier
            end

        println("Run once to account for compilation with $(typeof(clf)) ($(Dates.now()))")

        # Run once to account for compilation
        explain(X[first_neg, :], X, p, clf;
            desired_class=desired_class,
            compress_data=compress,
            min_num_generations=5,
            max_num_generations=5,
            convergence_k=3,
            ablation=true,
            run_crossover=crossover_run,
            run_mutation=mutation_run,
            domains=domains)

        println("Start process $num_to_explain instances ($(Dates.now()))")

        for i in 1:length(predictions)
            if predictions[i] != desired_class

                (i % 100 == 0) && println("$(@sprintf("%.2f", 100*num_explained/num_to_explain))% through .. ")

                orig_instance = X[i, :]

                clf =
                    if partial
                        if classifier isa PyCall.PyObject
                            initMLPEval(classifier, orig_instance)
                        else
                            initPartialRandomForestEval(classifier, orig_instance, 1)
                        end
                    else
                        classifier
                    end

                time = @elapsed (explanation, count, generation, rep_size, ptime, stime, mtime, ctime) =
                    explain(orig_instance, X, p, clf;
                        desired_class=desired_class,
                        compress_data=compress,
                        min_num_generations=5,
                        max_num_generations=5,
                        convergence_k=3,
                        ablation=true,
                        run_crossover=crossover_run,
                        run_mutation=mutation_run,
                        return_df=true,
                        domains=domains)

                # dist = distance(explanation[1:3, :], orig_instance, features, distance_temp; norm_ratio=[])
                for (fidx, feat) in enumerate(propertynames(X))
                    changed_feats[fidx] = (orig_instance[feat] != explanation[1,feat])
                end

                num_explained += 1

                ## We only consider the top-explanation for this
                correct_outcome[num_explained] = explanation[1,:outc]
                num_changed[num_explained] = sum(changed_feats)
                feat_changed[num_explained] = changed_feats
                distances[num_explained] = explanation.score[1]
                times[num_explained] = time
                num_generation[num_explained] = generation
                num_explored[num_explained] = count
                avg_rep_size[num_explained] = mean(rep_size[1:generation+1])

                prep_time[num_explained] = ptime
                selection_time[num_explained] = stime
                mutation_time[num_explained] = mtime
                crossover_time[num_explained] = ctime

                (num_explained >= num_to_explain) && break
            end
        end

        model_name = (partial ?  "$(model)_partial_model" : "$(model)_mlj_model")
        file = "scripts/results/ablation_exp/geco_ablation_experiments_$(dataset_name)_$(model_name)_compress_$(compress)_mutation_$(mutation_run)_crossover_$(crossover_run).jld"
        JLD.save(file,
            "times", times,
            "dist", distances,
            "numfeat", num_changed,
            "num_generation", num_generation,
            "num_explored", num_explored,
            "avg_rep_size", avg_rep_size,
            "prep_time", prep_time,
            "selection_time", selection_time,
            "mutation_time", mutation_time,
            "crossover_time", crossover_time)

        println("
            Average number of features changed: $(mean(num_changed))
            Average distances:                  $(mean(distances)) (normalized: $((mean(distances ./ size(X,2)))))
            Average times:                      $(mean(times)) ($(minimum(times)),$(maximum(times)))
            Average generations:                $(mean(num_generation))
            Average generated cfs:              $(mean(num_explored))
            Average representatino size:        $(mean(avg_rep_size))
            Correct outcomes:                   $(mean(correct_outcome))
            prep_time:                          $(mean(prep_time))  ($(minimum(prep_time)),$(maximum(prep_time)))
            selection_time:                     $(mean(selection_time))  ($(minimum(selection_time)),$(maximum(selection_time)))
            mutation_time:                      $(mean(mutation_time))  ($(minimum(mutation_time)),$(maximum(mutation_time)))
            crossover_time                      $(mean(crossover_time))  ($(minimum(crossover_time)),$(maximum(crossover_time)))
            Saved to: $file")
    end
end


for dataset in ["allstate", "yelp"]
    for model in ["PRF", "MLP"]

        include("../$(dataset)/$(dataset)_setup_$(model).jl")
        runBreakdownExperiment(X, p, classifier, dataset, model, 1)
    end
end
