using Plots, StatsPlots

import NPZ: npzread
import JLD

gc_file = wit_file = mace_file = ""

dataset = "credit"

# if dataset == "adult"
#     gc_file = "scripts/log/adult_gencount_with_mace_experiment_2020-10-19_0:28.jld"
#     wit_file = "scripts/log/adult_mo_with_mace_experiment_2020-10-19_0:54.jld"
#     mace_file = "scripts/log/result_adult_MACE.npy"
# elseif dataset == "credit"
#     gc_file = "scripts/log/credit_gencount_with_mace_experiment_2020-10-15_15:15.jld"
#     wit_file = "scripts/log/credit_mo_with_mace_experiment_2020-10-15_14:29.jld"
#     mace_file = "scripts/log/MACE_result.npy"
# elseif dataset == "compass"
#     gc_file = "scripts/log/compas_gencount_with_mace_experiment_2020-10-19_1:3.jld"
#     wit_file = "scripts/log/compas_mo_with_mace_experiment_2020-10-19_1:4.jld"
#     mace_file = "scripts/log/result_pass_MACE.npy"
# end

if dataset == "adult"
    gc_l1_file = "scripts/results/mace_exp/adult_geco_mace_experiment_ratio_l1_compress_false.jld"
    gc_l0l1_file = "scripts/results/mace_exp/adult_geco_mace_experiment_ratio_l0l1_compress_false.jld"
    wit_file = "scripts/results/moe_exp/adult_mo_mace_experiment_feasible_true.jld"
    mace_file = "scripts/results/mace_exp/mace_adult_tree_result.npy"
elseif dataset == "credit"
    gc_l1_file = "scripts/results/naive_exp/credit_geco_naive_experiment_ratio_l1_compress_false.jld"
    gc_l0l1_file = "scripts/results/naive_exp/credit_geco_naive_experiment_ratio_l0l1_compress_false.jld"
    naive_l1_file = "scripts/results/naive_exp/credit_gcNaive_naive_experiment_ratio_l1_compress_false.jld"
    naive_l0l1_file = "scripts/results/naive_exp/credit_gcNaive_naive_experiment_ratio_l0l1_compress_false.jld"
elseif dataset == "compass"
    gc_l1_file = "scripts/results/mace_exp/compas_geco_mace_experiment_ratio_l1_compress_false.jld"
    gc_l0l1_file = "scripts/results/mace_exp/compas_geco_mace_experiment_ratio_l0l1_compress_false.jld"
    wit_file = "scripts/results/moe_exp/compas_mo_mace_experiment_feasible_true.jld"
    mace_file = "scripts/results/mace_exp/mace_compas_tree_result.npy"
end

gc_dict = JLD.load(gc_l0l1_file)
gc_l1_dict = JLD.load(gc_l1_file)
naive_dict = JLD.load(naive_l0l1_file)
naive_l1_dict = JLD.load(naive_l1_file)

# default(titlefont = (20, "times"), legendfontsize = 18, guidefont = (18, :darkgreen), tickfont = (12, :orange), guide = "x", framestyle = :zerolines, yminorgrid = true)
default(titlefont = (20, "times"), guidefont = (16, :black), tickfont = (16, :black),)

times_plot = boxplot(["GeCo(l0+l1)" "GeCo(l1)" "Naive(l0+l1)" "Naive(l1)"],[gc_dict["times"] gc_l1_dict["times"] naive_dict["times"] naive_l1_dict["times"]], label="", ylabel="seconds",  yaxis=:log) #  title = "Time",
#times_plot = boxplot(["GeCo-l0+l1" "GeCo-l1" "WIT" "MACE"],[gc_dict["times"] gc_l1_dict["times"] mo_dict["times"] mace_array[1,:]], label="", title = "Time", ylabel="seconds")
savefig(times_plot, "scripts/plots/naive_exp_times_$dataset")

dist_plot = boxplot(["GeCo(l0+l1)" "GeCo(l1)" "Naive(l0+l1)" "Naive(l1)"],[gc_dict["dist"] gc_l1_dict["dist"] naive_dict["dist"] naive_l1_dict["dist"]], label="",  ylabel="l1-norm") # title = "Distance",
savefig(dist_plot, "scripts/plots/naive_exp_dist_$dataset")

# numfeat_plot = boxplot(["GeCo-l0+l1" "GeCo-l1" "WIT" "MACE"],[gc_dict["numfeat"] gc_l1_dict["numfeat"] mo_dict["numfeat"] mace_array[3,:]], label="", title = "Number of Features Changed", ylabel="# Features")
# savefig(numfeat_plot, "scripts/plots/mace_exp_feat_$dataset")

# dist_plot_2 = boxplot(["GeCo" "MACE"],[gc_dict["dist"] mace_array[2,:]], label="")
# savefig(dist_plot_2, "scripts/plots/mace_exp_dist_mace_gc_$dataset")

# times_plot_2 = boxplot(["GeCo" "WIT"],[gc_dict["times"] mo_dict["times"] ], label="")
# savefig(times_plot_2, "scripts/plots/mace_exp_times_mo_gc_$dataset")
