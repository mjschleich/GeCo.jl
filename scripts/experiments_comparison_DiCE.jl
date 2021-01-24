using Pkg; Pkg.activate(".")
using GeCo

using Printf, DataFrames
import Dates, JLD

function evalDiceExp(counterfactual, orig_instance)
    dis = 0
    num = 0
    for i in [[0], [1], [2,3,4,5],[6,7,8,9,10,11,12,13],[14,15,16,17,18],[19,20,21,22,23,24],[25,26],[27,28]]
        if (length(i) == 1 )
            if (counterfactual[i[1]+1] != orig_instance[i[1]+1])
                dis += abs(counterfactual[i[1]+1] - orig_instance[i[1]+1])
                num += 1
            end
        else
            for index in i
                if counterfactual[index+1] != orig_instance[index+1]
                    num += 1
                    dis += 1
                    break
                end
            end
        end
    end
    dis = dis/8
    return (dis, num)
end


function runDiceExperiment(X::DataFrame, p::PLAFProgram, dataset::String, desired_class::Int64)

    domains = initDomains(p,X)

    ## Use for the Torch comparison:
    in = torch.tensor(convert(Matrix, X)).float()
    predictions = classifier(in).float().detach().numpy()[:,desired_class+1]

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
    explain(X[1, :], X, path, classifier; desired_class = desired_class)

    for ratio in ["l0l1", "l1"] ## ["l0l1", "l1", "combined"]
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

                    (dis, num) = evalDiceExp(explanation[1,:], orig_instance)

                    ## We only consider the top-explanation for this
                    push!(correct_outcome, explanation[1,:outc])
                    push!(num_changed, num)
                    push!(distances, dis)
                    push!(times, time)
                    push!(num_generation, generation)
                    push!(num_explored, count)

                    num_explained += 1
                    (num_explained >= num_to_explain) && break
                end
            end

            #file = "scripts/results/mace_exp/$(dataset)_geco_mace_experiment_$(Dates.today())_$(Dates.hour(Dates.now())):$(Dates.minute(Dates.now())).jld"
            file = "scripts/results/dice_exp/$(dataset)_geco_dice_experiment_ratio_$(ratio)_compress_$(compress).jld"

            JLD.save(file, "times", times, "dist", distances, "numfeat", num_changed, "num_generation", num_generation, "num_explored", num_explored)

            println("
                Average number of features changed: $(mean(num_changed))
                Average distances:                  $(mean(distances)) (normalized: $((mean(distances ./ size(X,2)))))
                Average times:                      $(mean(times))
                Average generations:                $(mean(num_generation))
                Average generated cfs:              $(mean(num_explored))
                Correct outcomes:                   $(mean(correct_outcome))
                Saved to: $file")
        end
    end
end


include("adult/adult_setup_Dice.jl")

runDiceExperiment(X, p, "adult", 1)


