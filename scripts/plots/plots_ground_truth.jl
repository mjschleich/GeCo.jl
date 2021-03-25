using Plots, StatsPlots, Statistics, LaTeXStrings, Printf
using Plots.PlotMeasures
import JLD

# samples1 = (samples_mut = 5, samples_init = 20)
# samples2 = (samples_mut = 10, samples_init = 40)
# samples3 = (samples_mut = 100, samples_init = 300)

samples1 = (samples_mut = 5, samples_init = 20)
samples2 = (samples_mut = 15, samples_init = 60)
samples3 = (samples_mut = 25, samples_init = 100)
samples = [samples1, samples3] # [samples1,samples2]

norms = ["l1_norm"] # ["l1_norm", "l0_l1_norm"]
suffixes = ["interleaved_domain_size_no_max"] # ["decreasing_domain_size_no_max"] # ["_decreasing_domain_size","_interleaved_domain_size"]
exp_names = ["1features", "2features", "3features", "4features", "5features", "6features", "7features", "8features", "9features", "10features", "11features", "12features"]
exp_names_short = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

experiments = Dict{String,Dict}()
for exp_name in exp_names, suffix in suffixes, norm in norms,  num_samples in samples
    # file = "credit_ground_truth_experiment_symbols_$(exp_name)_ratio_l1_norm_samples_$(num_samples.samples_init)_$(num_samples.samples_mut).jld"
    file = "credit_ground_truth_exp_symbols_$(exp_name)_ratio_$(norm)_samples_$(num_samples.samples_init)_$(num_samples.samples_mut)_$suffix.jld"
    dict = JLD.load("scripts/results/ground_truth_exp/" * file)
    experiments[file] = dict
end


for suffix in suffixes, norm in norms

    times = Vector{Float64}()
    times_std = Vector{Float64}()
    gens =  Vector{Float64}()
    gens_std =  Vector{Float64}()

    distances_to_original = Vector{Float64}()
    distances_to_optimal = Vector{Float64}()
    distances_optimal_to_orig = Vector{Float64}()

    for exp_name in exp_names, num_samples in [samples1]

        file = "credit_ground_truth_exp_symbols_$(exp_name)_ratio_$(norm)_samples_$(num_samples.samples_init)_$(num_samples.samples_mut)_$suffix.jld"

        exp_time = experiments[file]["times"]
        exp_gens = experiments[file]["num_generation"]
        exp_dist_orig = experiments[file]["distances_to_original"]
        exp_dist_opt = experiments[file]["distances_to_optimal"]
        exp_dist_opt2orig = experiments[file]["distances_optimal_to_orig"]
        # exp_outc = experiments[file]["outc"]

        push!(times, mean(exp_time))
        push!(times_std, std(exp_time))

        push!(gens, mean(exp_gens))
        push!(gens_std, std(exp_gens))

        push!(distances_to_original, mean(exp_dist_orig))
        push!(distances_to_optimal, mean(exp_dist_opt))
        push!(distances_optimal_to_orig, mean(exp_dist_opt2orig))

        println("$exp_name   Dist_to_orig: $(distances_to_original[end]) Dist_to_opt: $(distances_to_optimal[end]) Dist_optimal_to_orig $(distances_optimal_to_orig[end])")
        println("            % Diff: $(100.0 * distances_to_optimal[end] / distances_to_original[end])")
    end

    times_large = Vector{Float64}()
    times_std_large = Vector{Float64}()
    gens_large =  Vector{Float64}()
    gens_std_large =  Vector{Float64}()

    distances_to_original_large = Vector{Float64}()
    distances_to_optimal_large = Vector{Float64}()
    distances_optimal_to_orig_large = Vector{Float64}()

    for exp_name in exp_names, num_samples in [samples3]
        file = "credit_ground_truth_exp_symbols_$(exp_name)_ratio_$(norm)_samples_$(num_samples.samples_init)_$(num_samples.samples_mut)_$suffix.jld"

        exp_time = experiments[file]["times"]
        exp_gens = experiments[file]["num_generation"]
        exp_dist_orig = experiments[file]["distances_to_original"]
        exp_dist_opt = experiments[file]["distances_to_optimal"]
        exp_dist_opt2orig = experiments[file]["distances_optimal_to_orig"]
        # exp_outc = experiments[file]["outc"]

        push!(times_large, mean(exp_time))
        push!(times_std_large, std(exp_time))

        push!(gens_large, mean(exp_gens))
        push!(gens_std_large, std(exp_gens))

        push!(distances_to_original_large, mean(exp_dist_orig))
        push!(distances_to_optimal_large, mean(exp_dist_opt))
        push!(distances_optimal_to_orig_large, mean(exp_dist_opt2orig))
    end


    default(guidefontsize=13, tickfontsize=12, legendfontsize=13)

    times_plot = plot(1:length(times), times,
        ylabel="Seconds",
        framestyle=:box,
        size=(400, 300),
        label="",
        ylims=(0, maximum(times) + 0.02),
        xticks=(1:length(times)),
        left_margin=0mm,
        right_margin=12mm,
        top_margin=0mm,
        bottom_margin=0mm,
        seriescolor=:blue,
        marker=3)

    # plot!(1:length(times), times_large,
    #     ylabel="",
    #     framestyle=:box,
    #     size=(400, 300),
    #     label="",
    #     ylims=(0, maximum(times_large) + 0.02),
    #     xticks=(1:length(times)),
    #     seriescolor=:green,
    #     marker=3)

    plot!(1:length(times), NaN .* (times),
        label="Runtime",
        grid=false,
        seriescolor=:blue,
        marker=3)

    plot!(1:length(times), NaN .* (times),
        framestyle=:box,
        label="Generations",
        grid=false,
        seriescolor=:red,
        legend=:topleft,
        marker=3)

    plot!(twinx(), gens,
        ylabel="Generations",
        framestyle=:box,
        legend=false,
        ylims=(0, maximum(gens) + 0.5),
        xticks=(1:length(times)),
        seriescolor=:red,
        marker=3)

    # plot!(twinx(), gens_large,
    #     ylabel="",
    #     framestyle=:box,
    #     legend=false,
    #     ylims=(0, maximum(gens_large) + 0.5),
    #     xticks=(1:length(times)),
    #     seriescolor=:red,
    #     marker=3)

    savefig(times_plot, "scripts/plots/ground_truth/ground_truth_times_credit_$(norm)_samples_20_5_$suffix")

    times_plot_large = plot(1:length(times), times,
        ylabel="Seconds",
        framestyle=:box,
        size=(600, 300),
        label="GeCo (default)",
        ylims=(0, maximum(times) + 0.02),
        xticks=(1:length(times)),
        legend=:topleft,
        seriescolor=:blue,
        margin=0mm,
        marker=3)

    plot!(1:length(times), times_large,
        framestyle=:box,
        label="GeCo (5x samples)",
        ylims=(0, maximum(times_large) + 0.02),
        xticks=(1:length(times)),
        seriescolor=:green,
        marker=3)

    savefig(times_plot_large, "scripts/plots/ground_truth/ground_truth_times_credit_$(norm)_samples_100_25_$suffix")

    dist_plot = groupedbar(1:length(distances_to_original), [distances_to_original distances_optimal_to_orig],
        bar_position=:doge,
        bar_width=0.5,
        xticks=(1:length(times)),
        ylabel="l1-norm",
        label=["GeCo" "Optimal Explanation"],
        size=(400, 300),
        legend=:topleft,
        framestyle=:box,
        margin=0mm,
        palette=:tab10
    )

    savefig(dist_plot, "scripts/plots/ground_truth/ground_truth_dist_credit_$(norm)_samples_20_5_$suffix")

    dist_plot_large = groupedbar(1:length(distances_to_original), [distances_to_original distances_to_original_large distances_optimal_to_orig],
        bar_position=:doge,
        bar_width=0.5,
        xticks=(1:length(times)),
        ylabel="l1-norm",
        label=["GeCo (default)" "GeCo (5x samples)" "Optimal Explanation"],
        size=(600, 300),
        legend=:topleft,
        framestyle=:box,
        margin=0mm,
        palette=:tab10
    )

    savefig(dist_plot_large, "scripts/plots/ground_truth/ground_truth_dist_credit_$(norm)_samples_100_25_$suffix")

end