using Plots, StatsPlots, Statistics, LaTeXStrings
using Plots.PlotMeasures

import NPZ: npzread
import JLD
using Printf

for dataset in ["credit", "adult"]

    if dataset == "adult"
        gc_l1_file = "scripts/results/mace_exp/adult_geco_mace_experiment_ratio_l1_compress_false.jld"
        gc_l0l1_file = "scripts/results/mace_exp/adult_geco_mace_experiment_ratio_l0l1_compress_false.jld"
        wit_file = "scripts/results/wit_exp/adult_wit_mace_experiment_feasible_true.jld"
        mace_file = "scripts/results/mace_exp/mace_adult_tree_result_constraint.npy"
        simba_file = "scripts/results/mace_exp/adult_simba_experiment.jld"
        cert_file = "scripts/results/naive_exp/adult_naive_ga_experiment_ratio_l1_generations_300.jld"
    elseif dataset == "credit"
        gc_l1_file = "scripts/results/mace_exp/credit_geco_mace_experiment_ratio_l1_compress_false.jld"
        gc_l0l1_file = "scripts/results/mace_exp/credit_geco_mace_experiment_ratio_l0l1_compress_false.jld"
        wit_file = "scripts/results/wit_exp/credit_wit_mace_experiment_feasible_true.jld"
        mace_file = "scripts/results/mace_exp/mace_credit_tree_result_constraint.npy"
        simba_file = "scripts/results/mace_exp/credit_simba_experiment.jld"
        cert_file = "scripts/results/naive_exp/credit_naive_ga_experiment_ratio_l1_generations_300.jld"
    # elseif dataset == "compass"
    #     gc_l1_file = "scripts/results/mace_exp/compas_geco_mace_experiment_ratio_l1_compress_false.jld"
    #     gc_l0l1_file = "scripts/results/mace_exp/compas_geco_mace_experiment_ratio_l0l1_compress_false.jld"
    #     wit_file = "scripts/results/wit_exp/compas_wit_mace_experiment_feasible_true.jld"
    #     mace_file = "scripts/results/mace_exp/mace_compas_tree_result.npy"
    end

    gc_dict = JLD.load(gc_l0l1_file)
    gc_l1_dict = JLD.load(gc_l1_file)
    wit_dict = JLD.load(wit_file)
    mace_array = npzread(mace_file)
    cert_dict = JLD.load(cert_file)
    simba_dict = JLD.load(simba_file)

    simba_correct_outc = simba_dict["desired_outc"]
    simba_times = simba_dict["times"][simba_correct_outc]
    simba_dist = simba_dict["dist"][simba_correct_outc]

    # default(titlefont = (20, "times"),legendfontsize = 18, guidefont = (18, :darkgreen), tickfont = (12, :orange), guide = "x", framestyle = :zerolines, yminorgrid = true)
    # default(titlefont = (20, "times"), guidefont = (12, :black), tickfont = (16, :black),)
    default(guidefontsize=13,tickfontsize=12,legendfontsize = 13)

    scale_factor = 1
    geco_times = copy(gc_l1_dict["times"][5001:end]) .* scale_factor
    wit_times = copy(wit_dict["times"]) .* scale_factor
    cert_times = copy(cert_dict["times"]) .* scale_factor
    simba_times = copy(simba_dict["times"][simba_correct_outc]) .* scale_factor
    mace_times = copy(mace_array[1,:]) .* scale_factor

    times_plot = bar([1 2 3 4 5],
        [mean(geco_times) mean(wit_times) mean(cert_times) mean(mace_times) mean(simba_times)],
        xticks=(1:5, ["GeCo" "WIT" "CERT" "MACE"  "SimCF"]),
        yerr=[std(geco_times) std(wit_times) std(cert_times) std(mace_times) std(simba_times)],
        ylabel="seconds",
        framestyle=:box,
        size=(400,300),
        label="",
        palette=:tab10)
        # palette=:lighttest) ## palette = :tab10

    if dataset == "adult"
        y_tiks = [0.0, 0.1, 0.2, 0.3]
    else
        y_tiks = [0.0, 0.1, 0.2]
    end

    bar!(times_plot, [1 2 3],
        [mean(geco_times) mean(wit_times) mean(simba_times)],
        xticks=(1:3, ["GeCo" "WIT" "SimCF"]),
        yticks=y_tiks, # [0.0, 0.1, 0.2],
        label="",
        yerr=[std(geco_times)  std(wit_times) std(simba_times)],
        inset=(1, bbox(0.12, 0.03, 0.43, 0.5, :top)),
        subplot = 2,
        margin = 0mm,
        # ylims=(-0.01,mean(geco_times)+0.02),
        framestyle = :box,
        tickfontsize=12,
        palette=[palette(:tab10)[1], palette(:tab10)[2], palette(:tab10)[5]])

    savefig(times_plot, "scripts/plots/mace_exp/mace_exp_times_$dataset")

    println("Speedup of GeCo over MACE: ", mean(mace_array[1,:])/mean(gc_l1_dict["times"][5001:end]))
    println("Speedup of GeCo over CERT: ", mean(cert_times)/mean(geco_times))
    println("SimBA: number of correct outcomes: $(sum(simba_correct_outc)) == $(length(simba_times))\n"*
            "       number of failed outc: $(length(simba_dict["times"]) - sum(simba_correct_outc))\n"*
            "       percentage failed: $(100.0 * (length(simba_dict["times"]) - sum(simba_correct_outc)) / length(simba_dict["times"]))\n"*
            "       mean times: $(mean(simba_times)) mean dist: $(mean(simba_dist))")

    # means = [mean(gc_l1_dict["dist"][5001:end]) mean(wit_dict["dist"]) mean(cert_dict["dist"]) mean(mace_array[2,:]) mean(simba_dist)]
    # stds = [std(gc_l1_dict["dist"][5001:end]) std(wit_dict["dist"]) std(cert_dict["dist"]) std(mace_array[2,:]) std(simba_dist)]
    # lower_err = [(s > m) ? (m,s) : (s,s) for (s,m) in zip(stds,means)]

    # ylims = if dataset == "credit"; (-0.0005, 0.008) else (-0.002, 0.075) end
    ylims = if dataset == "credit"; (-0.0005, 0.2) else (-0.002, 0.2) end

    geco_dist = copy(gc_l1_dict["dist"][5001:end])
    wit_dist = copy(wit_dict["dist"])
    cert_dist = copy(cert_dict["dist"])
    simba_dist = copy(simba_dict["dist"][simba_correct_outc])
    mace_dist = copy(mace_array[2,:])


    println("DISTANCES: ")
    println(([mean(geco_dist) mean(wit_dist) mean(cert_dist) mean(mace_dist) mean(simba_dist)]))
    println(([std(geco_dist) std(wit_dist) std(cert_dist) std(mace_dist) std(simba_dist)]))
    println(log10.([mean(geco_dist) mean(wit_dist) mean(cert_dist) mean(mace_dist) mean(simba_dist)]))

    if dataset == "adult"
        geco_dist .*= 37 / 12
        wit_dist  .*= 37 / 12
        cert_dist .*= 37 / 12
        simba_dist .*= 37 / 12

        scale_factor = 10_000

        geco_dist .*= scale_factor
        wit_dist .*= scale_factor
        cert_dist .*= scale_factor
        mace_dist .*= scale_factor
        simba_dist .*= scale_factor

        # y_label = "l1-norm * 1K"
        y_label = "l1-norm"
        y_tiks = (0:5, [L"10^{-4}" L"10^{-3}" L"10^{-2}" L"10^{-1}" L"10^0"])
    elseif dataset == "credit"

        scale_factor = 10_000
        geco_dist .*= scale_factor
        wit_dist .*= scale_factor
        cert_dist .*= scale_factor
        mace_dist .*= scale_factor
        simba_dist .*= scale_factor

        # y_label = "l1-norm * 10K"
        y_label = "l1-norm"
        y_tiks = (0:5, [L"10^{-4}" L"10^{-3}" L"10^{-2}" L"10^{-1}" L"10^0"])
    end

    println(log10.([mean(geco_dist) mean(wit_dist) mean(cert_dist) mean(mace_dist) mean(simba_dist)]))
    println(log10.([std(geco_dist) std(wit_dist) std(cert_dist) std(mace_dist) std(simba_dist)]))

    dist_plot = bar([1 2 3 4 5],
        log10.([mean(geco_dist) mean(wit_dist) mean(cert_dist) mean(mace_dist) mean(simba_dist)]),
        xticks=(1:5, ["GeCo" "WIT" "CERT" "MACE" "SimCF"]),
        # yerr=log10.([std(geco_dist) std(wit_dist) std(cert_dist) std(mace_dist) std(simba_dist)]),
        ylabel=y_label,   ##"l1-norm",
        label="",
        margin=0mm,
        top_margin=1mm,
        size=(400,300),
        ylims=(-0.05, 4),
        yticks=y_tiks,
        # yticks=(0:5, [L"10^{0}" L"10^{1}" L"10^{2}" L"10^{3}" L"10^{4}"]),
        # yticks=(0:5, [L"10^{-4}" L"10^{-3}" L"10^{-2}" L"10^{-1}" L"10^0"]),
        # yaxis=(:log10, (0.00001,Inf)),
        framestyle = :box,
        palette=:tab10) # palette=:lighttest ## palette = :tab10

    # if dataset == "adult"
    #     annotate!(1.5,0.07,text("up to $(@sprintf("%.2f", (mean(wit_dist) + std(wit_dist)))) ↑"))
    #     annotate!(3.5,0.07,text("↑ up to $(@sprintf("%.2f", (mean(cert_dist) + std(cert_dist))))"))
    # end
    # if dataset == "credit"
    #     annotate!(1.5,0.0075,text("up to $(@sprintf("%.2f", (mean(wit_dist) + std(wit_dist)))) ↑"))
    # end

    savefig(dist_plot, "scripts/plots/mace_exp/mace_exp_dist_$dataset")

    println("$(dataset) - dist: $(mean(cert_dict["dist"])) times: $(mean(cert_dict["times"]))   ", mean(cert_dict["dist"])/mean(gc_l1_dict["dist"][5001:end]))

    numfeat_plot = bar([1 2 3 4],
        [mean(gc_dict["numfeat"]) mean(gc_l1_dict["numfeat"][5001:end]) mean(wit_dict["numfeat"]) mean(mace_array[3,:])],
        xticks=(1:4, ["GeCo" "GeCo (l1)" "wit" "cert" "MACE"]),
        yerr=[std(gc_dict["numfeat"])  std(gc_l1_dict["numfeat"][5001:end]) std(wit_dict["numfeat"]) std(mace_array[3,:])],
        ylabel="num features",
        label="",
        framestyle = :box,
        palette=:tab10) # palette=:lighttest) ## palette = :tab10
    savefig(numfeat_plot, "scripts/plots/mace_exp/mace_exp_feat_$dataset")

    wit_num_feat = copy(wit_dict["numfeat"])
    cert_num_feat = copy(cert_dict["numfeat"])
    simcf_num_feat = copy(simba_dict["numfeat"][simba_correct_outc])

    println("Number of features changed: geco $(mean(gc_l1_dict["numfeat"][5001:end]))  WIT $(mean(wit_dict["numfeat"])) cert $(mean(cert_dict["numfeat"])) MACE $(mean(mace_array[3,:]))")
    println("WIT: Number of times without explanation: $(5000 - length(wit_num_feat)) or $(100.0*(5000 - length(wit_num_feat))/5000)% ")
    println("CERT: Number of times without explanation: $(5000 - length(cert_num_feat)) or $(100.0*(5000 - length(cert_num_feat))/5000)% ")

    for i in length(wit_num_feat):5000
        push!(wit_num_feat, 10)
    end
    for i in length(cert_num_feat):5000
        push!(cert_num_feat, 10)
    end
    for i in length(simcf_num_feat):5000
        push!(simcf_num_feat, 10)
    end

    groups = [["GeCo" for _ in 1:5000]; ["GeCo (l1)" for _ in 1:5000]; ["wit" for _ in 1:length(wit_num_feat)]; ["cert" for _ in 1:length(cert_num_feat)]; ["MACE" for _ in 1:5000]; ]
    numfeat_plot = groupedhist(
        [gc_dict["numfeat"]; gc_l1_dict["numfeat"][5001:end]; wit_num_feat; cert_num_feat; mace_array[3,:]],
        group=groups,
        xticks=(1:10, [string.(i for i in 1:9); ["∞"]]),
        bar_width=.7,
        bar_position = :doge,
        ylabel="Frequency",
        # xlabel="Number of features changed",
        framestyle = :box)
    savefig(numfeat_plot, "scripts/plots/mace_exp/mace_exp_feat_$(dataset)_hist")

    colors = [RGB{Float64}(0.1216,0.4667,0.7059), RGB{Float64}(1.0,0.498,0.0549), RGB{Float64}(0.1725,0.6275,0.1725), RGB{Float64}(0.8392,0.1529,0.1569), RGB{Float64}(0.5804,0.4039,0.7412)]

    groups_wit = [["GeCo" for _ in 1:5000]; ["WIT" for _ in 1:length(wit_num_feat)]]

    numfeat_plot = groupedhist(
        [gc_l1_dict["numfeat"][5001:end]; wit_num_feat],
        group=groups_wit,
        xticks=(1:10, [string.(i for i in 1:9); ["∞"]]),
        xrange=(0.5,10.8),
        yrange=(-10,4000),
        bar_width=.7,
        bar_position = :doge,
        color = [colors[1] colors[2]],
        # ylabel="Frequency",
        framestyle = :box,
        size=(500,200))

    for i in 1:maximum(wit_num_feat)
        freq = sum(wit_num_feat .== i)
        if freq <= 1; continue; end
        if i == 1
            annotate!(i+0.5, freq+300, text("$(freq)", 10))
        elseif i == 2 && dataset == "credit"
            annotate!(i+0.54, freq+300, text("$(freq)", 10))
        else
            annotate!(i+0.4, freq+300, text("$(freq)", 10))
        end
    end

    savefig(numfeat_plot, "scripts/plots/mace_exp/mace_exp_feat_$(dataset)_hist_wit")

    groups_cert = [["GeCo" for _ in 1:5000]; ["xCERT" for _ in 1:length(cert_num_feat)] ]
    numfeat_plot = groupedhist(
        [gc_l1_dict["numfeat"][5001:end]; cert_num_feat],
        group=groups_cert,
        xticks=(1:10, [string.(i for i in 1:9); ["∞"]]),
        xrange=(0.5,10.8),
        bar_width=.7,
        label=["GeCo" "CERT"],
        color = [colors[1] colors[3]],
        bar_position = :doge,
        framestyle = :box,
        size=(500,200))

    for i in 1:maximum(cert_num_feat)
        freq = sum(cert_num_feat .== i)
        if freq <= 1; continue; end

        if i < 3
            annotate!(i+0.43, freq+300, text("$(freq)", 10))
        else
            annotate!(i+0.2, freq+300, text("$(freq)", 10))
        end
    end

    savefig(numfeat_plot, "scripts/plots/mace_exp/mace_exp_feat_$(dataset)_hist_cert")

    groups_simcf = [["GeCo" for _ in 1:5000]; ["xsimcf" for _ in 1:length(simcf_num_feat)] ]
    numfeat_plot = groupedhist(
        [gc_l1_dict["numfeat"][5001:end]; simcf_num_feat],
        group=groups_simcf,
        xticks=(1:10, [string.(i for i in 1:9); ["∞"]]),
        xrange=(0.5,10.8),
        bar_width=.7,
        label=["GeCo" "SimCF"],
        color = [colors[1] colors[5]],
        bar_position = :doge,
        framestyle = :box,
        size=(500,200))

    for i in 1:maximum(simcf_num_feat)
        freq = sum(simcf_num_feat .== i)
        if freq <= 1; continue; end

        if i < 3
            annotate!(i+0.43, freq+300, text("$(freq)", 10))
        else
            annotate!(i+0.2, freq+300, text("$(freq)", 10))
        end
    end

    savefig(numfeat_plot, "scripts/plots/mace_exp/mace_exp_feat_$(dataset)_hist_simcf")


    groups_mace = [["GeCo" for _ in 1:5000]; ["MACE" for _ in 1:5000] ]
    numfeat_plot = groupedhist(
        [gc_l1_dict["numfeat"][5001:end]; mace_array[3,:]],
        group=groups_mace,
        xticks=(1:10, [string.(i for i in 1:9); ["∞"]]),
        bar_width=.7,
        xrange=(0.5,10.8),
        color = [colors[1] colors[4]],
        bar_position = :doge,
        framestyle = :box,
        size=(500,200))

    for i in 1:maximum(gc_l1_dict["numfeat"][5001:end])
        freq = sum(gc_l1_dict["numfeat"][5001:end] .== i)
        if dataset == "credit"
            if i == 1
                annotate!(i+0.7, freq-100, text("$(freq)",10,colors[1]))
            elseif i == 3
                annotate!(i-0.05, freq+200, text("$(freq)", 10,colors[1]))
            else
                annotate!(i, freq+200, text("$(freq)", 10,colors[1]))
            end
        else
            if i == 1
                annotate!(i+0.6, freq-100, text("$(freq)",10,colors[1]))
            elseif i == 2
                annotate!(i-0.25, freq+200, text("$(freq)",9,colors[1]))
            elseif i == 3
                annotate!(i-0.25, freq+200, text("$(freq)",10,colors[1]))
            else
                annotate!(i-0.1, freq+200, text("$(freq)",10,colors[1]))
            end
        end
    end

    for i in 1:maximum(mace_array[3,:])
        freq = sum(mace_array[3,:] .== i)

        if dataset== "credit"
            if freq < 1000
                annotate!(i+0.6, freq+300, text("$(freq)", 10))
            elseif i == 1
                annotate!(i+0.45, freq+300, text("$(freq)", 10))
            elseif i == 3
                annotate!(i+0.45, freq+350, text("$(freq)", 10))
            else
                annotate!(i+0.4, freq+300, text("$(freq)", 10))
            end
        end

        if dataset== "adult"
            if i ==1
                annotate!(i+0.45, freq+300, text("$(freq)", 10))
            elseif i == 3
                annotate!(i+0.3, freq+400, text("$(freq)", 10))
            else
                annotate!(i+0.4, freq+300, text("$(freq)", 10))
            end
        end
    end

    savefig(numfeat_plot, "scripts/plots/mace_exp/mace_exp_feat_$(dataset)_hist_mace")
end
