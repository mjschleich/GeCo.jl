using JLD, Plots, StatsPlots, NPZ, Statistics

function plotDiceData()
    path = "scripts/results/dice_exp/"

    gc_file = path*"adult_geco_dice_experiment_ratio_l1_compress_false.jld"
    gc_l1_file = path*"adult_geco_dice_experiment_ratio_l1_compress_false.jld"
    Dice_file = path*"dice_result.npy"

    gc_dict = JLD.load(gc_file)
    gc_l1_dict = JLD.load(gc_l1_file)
    dice_array = npzread(Dice_file)

    default(guidefontsize=12, tickfontsize=12, legendfontsize=11)

    times_plot = bar([1 2], [mean(gc_l1_dict["times"])*1000 mean(dice_array[1,:])*1000], xticks=(1:2, ["GeCo" "DiCE"]), yerr=[std(gc_dict["times"].*1000) std(dice_array[1,:].*1000)],  ylabel="milliseconds",  label="", framestyle = :box, palette=:lighttest) ## palette = :tab10
    savefig(times_plot, "scripts/plots/dice_exp_times")

    dist_plot = bar([1 2],
        [mean(gc_l1_dict["dist"])  mean(dice_array[2,:])],
        xticks=(1:2, ["GeCo" "DiCE"]),
        yerr=[std(gc_dict["dist"]) std(dice_array[2,:])],
        ylabel="l1-norm",
        label="",
        framestyle = :box,
        palette=:tab10,
        size=(300,200))
    savefig(dist_plot, "scripts/plots/dice_exp_dist")

    numfeat_plot = bar([1 2], [mean(gc_l1_dict["numfeat"])  mean(dice_array[3,:])], xticks=(1:2, ["GeCo" "DiCE"]), yerr=[std(gc_dict["numfeat"]) std(dice_array[3,:])],  ylabel="# features",  label="", framestyle = :box, palette=:lighttest) ## palette = :tab10
    savefig(numfeat_plot, "scripts/plots/dice_exp_feat")

    groups = [["GeCo" for _ in 1:5000]; ["xDiCE" for _ in 1:5000]; ]
    numfeat_plot = groupedhist(
        [gc_l1_dict["numfeat"]; dice_array[3,:]],
        group=groups,
        xticks=(1:8),
         # , [string.(i for i in 1:8)]),
        xrange=(0.6,9.5),
        bar_width=.7,
        bar_position = :doge,
        label=["GeCo" "DiCE"],
        size=(300,200),
        palette=:tab10,
        framestyle = :box)

    savefig(numfeat_plot, "scripts/plots/output/dice_exp_feat_hist")

end

plotDiceData()

# times_plot = boxplot(["GenCount_time"  "Dice_time"],[gc_dict["times"]  mace_array[1,:]], label="Time")
# savefig(times_plot, "scripts/plots/Dice_exp_times")

# dist_plot = boxplot(["GenCount_dist" "Dice_dist"],[gc_dict["dist"] mace_array[2,:]], label="")
# savefig(dist_plot, "scripts/plots/Dice_exp_dist")

# numfeat_plot = boxplot(["GenCount_numfeat" "Dice_numfeat"],[gc_dict["numfeat"]  mace_array[3,:]], label="Number of Feature Changed")
# savefig(numfeat_plot, "scripts/plots/Dice_exp_feat")