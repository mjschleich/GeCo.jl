using JLD, Plots, StatsPlots, NPZ, Statistics
using Plots.PlotMeasures

function plotDiceData()
    path = "scripts/results/dice_exp/"

    gc_file = path*"adult_geco_dice_experiment_ratio_l1_compress_false.jld"
    gc_l1_file = path*"adult_geco_dice_experiment_ratio_l1_compress_false.jld"
    Dice_file = path*"dice_result.npy"

    gc_dict = JLD.load(gc_file)
    gc_l1_dict = JLD.load(gc_l1_file)
    dice_array = npzread(Dice_file)

    default(guidefontsize=12, tickfontsize=12, legendfontsize=11)

    dist_plot = bar([1 2],
        [mean(gc_l1_dict["dist"])  mean(dice_array[2,:])],
        xticks=(1:2, ["GeCo" "DiCE"]),
        yerr=[std(gc_dict["dist"]) std(dice_array[2,:])],
        ylabel="l1-norm",
        label="",
        framestyle = :box,
        palette=:tab10,
        margin=0mm,
        size=(400,300)
        )
    savefig(dist_plot, "scripts/plots/dice_exp/dice_exp_dist")

    # numfeat_plot = bar([1 2], [mean(gc_l1_dict["numfeat"])  mean(dice_array[3,:])], xticks=(1:2, ["GeCo" "DiCE"]), yerr=[std(gc_dict["numfeat"]) std(dice_array[3,:])],  ylabel="# features",  label="", framestyle = :box, palette=:lighttest) ## palette = :tab10

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
        size=(500,200),
        margin=0mm,
        palette=:tab10,
        framestyle = :box)

    savefig(numfeat_plot, "scripts/plots/dice_exp/dice_exp_feat_hist")


    println("Geco Time: $(mean(gc_l1_dict["times"])*1000) Dice Time: $(mean(dice_array[1,:])*1000)")

    times_plot = bar([1 2],
        [mean(gc_l1_dict["times"]) mean(dice_array[1,:])],
        xticks=(1:2, ["GeCo" "DiCE"]),
        yerr=[std(gc_dict["times"]) std(dice_array[1,:])],
        ylabel="seconds",
        label="",
        margin=0mm,
        size=(400,300),
        framestyle = :box,
        palette=:tab10)

    savefig(times_plot, "scripts/plots/dice_exp/dice_exp_times")

    # times_plot = bar([1 2],
    #     [mean(gc_l1_dict["times"])  mean(dice_array[1,:])],
    #     xticks=(1:2, ["GeCo" "DiCE"]),
    #     yerr=[std(gc_l1_dict["times"])  std(dice_array[1,:])],
    #     ylabel="seconds",
    #     framestyle=:box,
    #     size=(300,200),
    #     margin=0mm,
    #     label="",
    #     palette=:tab10)

    # savefig(times_plot, "scripts/plots/dice_exp/dice_exp_times")

end

plotDiceData()

# times_plot = boxplot(["GenCount_time"  "Dice_time"],[gc_dict["times"]  mace_array[1,:]], label="Time")
# savefig(times_plot, "scripts/plots/Dice_exp_times")

# dist_plot = boxplot(["GenCount_dist" "Dice_dist"],[gc_dict["dist"] mace_array[2,:]], label="")
# savefig(dist_plot, "scripts/plots/Dice_exp_dist")

# numfeat_plot = boxplot(["GenCount_numfeat" "Dice_numfeat"],[gc_dict["numfeat"]  mace_array[3,:]], label="Number of Feature Changed")
# savefig(numfeat_plot, "scripts/plots/Dice_exp_feat")