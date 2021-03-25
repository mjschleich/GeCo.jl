using JLD, Plots, StatsPlots, NPZ, Statistics
using Plots.PlotMeasures

const compute_allgens = false

for dataset in ["yelp_MLP", "yelp_MLP_small", "yelp", "allstate"], use_allgens in [false ]      ##  ["yelp", "allstate", "yelp_MLP"]
        if dataset == "allstate"
                if !use_allgens
                        file_mlj_reg = "scripts/results/ablation_exp/allstate/geco_ablation_experiments_allstate_PRF_mlj_model_compress_false_mutation_true_crossover_true.jld"
                        file_mlj_comp ="scripts/results/ablation_exp/allstate/geco_ablation_experiments_allstate_PRF_mlj_model_compress_true_mutation_true_crossover_true.jld"
                        file_partial_reg = "scripts/results/ablation_exp/allstate/geco_ablation_experiments_allstate_PRF_partial_model_compress_false_mutation_true_crossover_true.jld"
                        file_partial_comp = "scripts/results/ablation_exp/allstate/geco_ablation_experiments_allstate_PRF_partial_model_compress_true_mutation_true_crossover_true.jld"
                else
                        file_mlj_reg = "scripts/results/ablation_exp/allgens/geco_ablation_experiments_allstate_PRF_mlj_model_compress_false_mutation_true_crossover_true_allgens.jld"
                        file_mlj_comp ="scripts/results/ablation_exp/allgens/geco_ablation_experiments_allstate_PRF_mlj_model_compress_true_mutation_true_crossover_true_allgens.jld"
                        file_partial_reg = "scripts/results/ablation_exp/allgens/geco_ablation_experiments_allstate_PRF_partial_model_compress_false_mutation_true_crossover_true_allgens.jld"
                        file_partial_comp = "scripts/results/ablation_exp/allgens/geco_ablation_experiments_allstate_PRF_partial_model_compress_true_mutation_true_crossover_true_allgens.jld"
                end
        elseif dataset == "allstate_MLP"
                file_mlj_reg = "scripts/results/ablation_exp/allstate/geco_ablation_experiments_allstate_MLP_mlj_model_compress_false_mutation_true_crossover_true.jld"
                file_mlj_comp ="scripts/results/ablation_exp/allstate/geco_ablation_experiments_allstate_MLP_mlj_model_compress_true_mutation_true_crossover_true.jld"
                file_partial_reg = "scripts/results/ablation_exp/allstate/geco_ablation_experiments_allstate_MLP_partial_model_compress_false_mutation_true_crossover_true.jld"
                file_partial_comp = "scripts/results/ablation_exp/allstate/geco_ablation_experiments_allstate_MLP_partial_model_compress_true_mutation_true_crossover_true.jld"
        elseif dataset == "yelp"
                if !use_allgens
                        file_mlj_reg = "scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_PRF_mlj_model_compress_false_mutation_true_crossover_true_less_categ.jld"
                        file_mlj_comp ="scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_PRF_mlj_model_compress_true_mutation_true_crossover_true_less_categ.jld"
                        file_partial_reg = "scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_PRF_partial_model_compress_false_mutation_true_crossover_true_less_categ.jld"
                        file_partial_comp = "scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_PRF_partial_model_compress_true_mutation_true_crossover_true_less_categ.jld"
                else
                        file_mlj_reg = "scripts/results/ablation_exp/allgens/geco_ablation_experiments_yelp_PRF_mlj_model_compress_false_mutation_true_crossover_true_allgens.jld"
                        file_mlj_comp ="scripts/results/ablation_exp/allgens/geco_ablation_experiments_yelp_PRF_mlj_model_compress_true_mutation_true_crossover_true_allgens.jld"
                        file_partial_reg = "scripts/results/ablation_exp/allgens/geco_ablation_experiments_yelp_PRF_partial_model_compress_false_mutation_true_crossover_true_allgens.jld"
                        file_partial_comp = "scripts/results/ablation_exp/allgens/geco_ablation_experiments_yelp_PRF_partial_model_compress_true_mutation_true_crossover_true_allgens.jld"
                end
        elseif dataset == "yelp_MLP"
                if !use_allgens
                        file_mlj_reg = "scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_MLP_mlj_model_compress_false_mutation_true_crossover_true_less_categ.jld"
                        file_mlj_comp ="scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_MLP_mlj_model_compress_true_mutation_true_crossover_true_less_categ.jld"
                        file_partial_reg = "scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_MLP_partial_model_compress_false_mutation_true_crossover_true_less_categ.jld"
                        file_partial_comp = "scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_MLP_partial_model_compress_true_mutation_true_crossover_true_less_categ.jld"
                else
                        file_mlj_reg = "scripts/results/ablation_exp/allgens/geco_ablation_experiments_yelp_MLP_mlj_model_compress_false_mutation_true_crossover_true_allgens.jld"
                        file_mlj_comp ="scripts/results/ablation_exp/allgens/geco_ablation_experiments_yelp_MLP_mlj_model_compress_true_mutation_true_crossover_true_allgens.jld"
                        file_partial_reg = "scripts/results/ablation_exp/allgens/geco_ablation_experiments_yelp_MLP_partial_model_compress_false_mutation_true_crossover_true_allgens.jld"
                        file_partial_comp = "scripts/results/ablation_exp/allgens/geco_ablation_experiments_yelp_MLP_partial_model_compress_true_mutation_true_crossover_true_allgens.jld"
                end
        elseif dataset == "yelp_MLP_small"
                file_mlj_reg = "scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_MLP_mlj_model_compress_false_mutation_true_crossover_true_less_categ_small.jld"
                file_mlj_comp ="scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_MLP_mlj_model_compress_true_mutation_true_crossover_true_less_categ_small.jld"
                file_partial_reg = "scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_MLP_partial_model_compress_false_mutation_true_crossover_true_less_categ_small.jld"
                file_partial_comp = "scripts/results/ablation_exp/yelp/geco_ablation_experiments_yelp_MLP_partial_model_compress_true_mutation_true_crossover_true_less_categ_small.jld"
        end

        # default(guidefontsize=16, )
        default(guidefontsize=13,tickfontsize=12,legendfontsize = 13)

        mlj_reg = JLD.load(file_mlj_reg)
        mlj_comp = JLD.load(file_mlj_comp)

        partial_reg = JLD.load(file_partial_reg)
        partial_comp = JLD.load(file_partial_comp)

        println("$dataset  -- allgens: $use_allgens")
        println("       Number of generations: $(mean(mlj_reg["num_generation"])) Number of explored CFs: $(mean(mlj_reg["num_explored"]))")
        println("       Avg Size Naive: $(mean(mlj_reg["avg_rep_size"])) Avg Size Compressed: $(mean(mlj_comp["avg_rep_size"])) Compression: $(mean(mlj_reg["avg_rep_size"]) / mean(mlj_comp["avg_rep_size"])) \n")

        println("       Overall Speedup: $(mean(mlj_reg["times"])/mean(partial_comp["times"]))  Only partial eval: $(mean(mlj_reg["times"])/mean(partial_reg["times"]))  Only comp: $(mean(mlj_reg["times"])/mean(mlj_comp["times"])) ")
        println("       Speedup Selection: $(mean(mlj_reg["selection_time"])/mean(partial_comp["selection_time"]))  Only partial eval: $(mean(mlj_reg["selection_time"])/mean(partial_reg["selection_time"]))  Only comp: $(mean(mlj_comp["selection_time"])/mean(mlj_reg["selection_time"])) SLOWER ")
        println("       Speedup Mutation : $(mean(mlj_reg["mutation_time"])/mean(partial_comp["mutation_time"]))  Only partial eval: $(mean(mlj_reg["mutation_time"])/mean(partial_reg["mutation_time"]))  Only comp: $(mean(mlj_reg["mutation_time"])/mean(mlj_comp["mutation_time"])) ")

        prep = [mean(mlj_reg["prep_time"]), mean(partial_reg["prep_time"]), mean(mlj_comp["prep_time"]), mean(partial_comp["prep_time"])]
        selection = [mean(mlj_reg["selection_time"]), mean(partial_reg["selection_time"]), mean(mlj_comp["selection_time"]), mean(partial_comp["selection_time"])]
        mutation = [mean(mlj_reg["mutation_time"]), mean(partial_reg["mutation_time"]), mean(mlj_comp["mutation_time"]), mean(partial_comp["mutation_time"])]
        crossover = [mean(mlj_reg["crossover_time"]), mean(partial_reg["crossover_time"]), mean(mlj_comp["crossover_time"]), mean(partial_comp["crossover_time"])]

        breakdown_plot = groupedbar([selection crossover mutation prep],
                bar_position = :stack,
                bar_width=0.5,
                xticks=(1:4, ["No Opt" "PartEval" "Δ-Rep" "PartEval+\nΔ-Rep"]),
                # label=["select" "crossover" "mutate"  "initial"],
                label="",
                tickfontsize=13,
                # legend=:top,
                ylabel="seconds",
                size=(400,300),
                margin=0mm,
                legendfontsize=13,
                framestyle = :box, # palette=:tab10
                )

        if !use_allgens
                savefig(breakdown_plot, "scripts/plots/breakdown_exp_$(dataset)_times")
        else
                savefig(breakdown_plot, "scripts/plots/output/breakdown_exp_$(dataset)_times_allgens")
        end

        println(keys(mlj_reg))

        # if compute_allgens
        #         mlj_reg = JLD.load(file_mlj_reg_2)
        #         mlj_comp = JLD.load(file_mlj_comp_2)
        #         partial_reg = JLD.load(file_partial_reg_2)
        #         partial_comp = JLD.load(file_partial_comp_2)
        #         println("(all gens) -- Number of generations: $(mean(mlj_reg["num_generation"])) Number of explored CFs: $(mean(mlj_reg["num_explored"]))")
        #         println("           -- Avg Size Naive: $(mean(mlj_reg["avg_rep_size"])) Avg Size Compressed: $(mean(mlj_comp["avg_rep_size"])) Compression: $(mean(mlj_reg["avg_rep_size"]) / mean(mlj_comp["avg_rep_size"])) \n")
        #         prep = [mean(mlj_reg["prep_time"]), mean(partial_reg["prep_time"]), mean(mlj_comp["prep_time"]), mean(partial_comp["prep_time"])]
        #         selection = [mean(mlj_reg["selection_time"]), mean(partial_reg["selection_time"]), mean(mlj_comp["selection_time"]), mean(partial_comp["selection_time"])]
        #         mutation = [mean(mlj_reg["mutation_time"]), mean(partial_reg["mutation_time"]), mean(mlj_comp["mutation_time"]), mean(partial_comp["mutation_time"])]
        #         crossover = [mean(mlj_reg["crossover_time"]), mean(partial_reg["crossover_time"]), mean(mlj_comp["crossover_time"]), mean(partial_comp["crossover_time"])]
        #         breakdown_plot = groupedbar([selection crossover mutation prep],
        #                 bar_position = :stack,
        #                 bar_width=0.5,
        #                 xticks=(1:4, ["No Opt." "Partial Comp." "Δ-Rep." "Partial Comp.+\nΔ-Rep."]),
        #                 label=["selection" "crossover" "mutation"  "preprocess"],
        #                 tickfontsize=12,
        #                 # legend=:top,
        #                 ylabel="seconds",
        #                 size=(550,400),
        #                 legendfontsize = 12,
        #                 framestyle = :box)
        #         savefig(breakdown_plot, "scripts/plots/output/breakdown_exp_$(dataset)_times_allgens")
        # end
end