### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 14adad40-6af7-4389-ae3a-b0d2a1462424
begin
	using GeCo, PlutoUI, DataFrames, Plots, MLJ, StatsPlots

	plotly()

	include("scripts/fico/fico_machine.jl");
	md"""
	**Step 1.** Provide a pre-learned classifier and dataset
	"""
end


# ╔═╡ baadbc4c-7ba3-11eb-3cbb-a32fc7755748
md""" 
## Demonstration of GeCo
"""



# ╔═╡ 1b2d0f6b-aab0-430f-9d92-67c6e585bc73
classifier.model

# ╔═╡ 3ebfd05d-8fac-4d4f-af0e-ba5d702014fa
X[1:3,:]

# ╔═╡ 4a44f2a8-a75a-11eb-2d89-3fda270ecac3
md""" 
**Step 2.** Define PLAF constraints to ensure plausibility and feasibilty of the explanations. """

# ╔═╡ 4e931614-a75a-11eb-3cbf-eff5cda26330
begin
	plaf_prog = PLAFProgram()

# 	@PLAF(p, :cf.ExternalRiskEstimate .>= :x.ExternalRiskEstimate)
# 	@PLAF(p, :cf.MSinceOldestTradeOpen .>= :x.MSinceOldestTradeOpen)
# 	@PLAF(p, :cf.MSinceMostRecentTradeOpen .>= :x.MSinceMostRecentTradeOpen)
# 	@PLAF(p, :cf.AverageMInFile .>= :x.AverageMInFile)
# 	@PLAF(p, :cf.PercentTradesNeverDelq .>= :x.PercentTradesNeverDelq)
# 	@PLAF(p, :cf.MSinceMostRecentDelq .>= :x.MSinceMostRecentDelq)
# 	@PLAF(p, :cf.MaxDelq2PublicRecLast12M .>= :x.MaxDelq2PublicRecLast12M)
# 	@PLAF(p, :cf.MaxDelqEver .>= :x.MaxDelqEver)
# 	@PLAF(p, :cf.NumTotalTrades .>= :x.NumTotalTrades)
# 	@PLAF(p, :cf.NumTradesOpeninLast12M .>= :x.NumTradesOpeninLast12M)
# 	@PLAF(p, :cf.MSinceMostRecentInqexcl7days .>= :x.MSinceMostRecentInqexcl7days)
# 	@PLAF(p, :cf.NumTradesOpeninLast12M .>= :x.NumTradesOpeninLast12M)
# 	@PLAF(p, :cf.NumTradesOpeninLast12M .>= :x.NumTradesOpeninLast12M)
# 	@PLAF(p, :cf.NumTradesOpeninLast12M .>= :x.NumTradesOpeninLast12M)
# 	@PLAF(p, :cf.NumTradesOpeninLast12M .>= :x.NumTradesOpeninLast12M)


# 	@PLAF(p, :cf.NumTrades60Ever2DerogPubRec .<= :x.NumTrades60Ever2DerogPubRec)
# 	@PLAF(p, :cf.PercentInstallTrades .<= :x.PercentInstallTrades)
# 	@PLAF(p, :cf.NumInqLast6M .<= :x.NumInqLast6M)
# 	@PLAF(p, :cf.NumInqLast6Mexcl7days .<= :x.NumInqLast6Mexcl7days)
# 	@PLAF(p, :cf.NetFractionRevolvingBurden .<= :x.NetFractionRevolvingBurden)
# 	@PLAF(p, :cf.NetFractionInstallBurden .<= :x.NetFractionInstallBurden)
# 	@PLAF(p, :cf.NumBank2NatlTradesWHighUtilization .<= :x.NumBank2NatlTradesWHighUtilization)
# 	@PLAF(p, :cf.PercentTradesWBalance .<= :x.PercentTradesWBalance)
end

# ╔═╡ 08034110-b5a7-44c6-8a71-725998b43f60
md""" 
**Step 3.** Define the instance to be explained """

# ╔═╡ 2a1fb272-4a2d-45c6-ba58-794fe3012c15
md"""
| ExternalRiskEstimate | MSinceOldestTradeOpen | MSinceMostRecentTradeOpen | AverageMInFile | NumSatisfactoryTrades | NumTrades60Ever2DerogPubRec | NumTrades90Ever2DerogPubRec | PercentTradesNeverDelq | MSinceMostRecentDelq | MaxDelq2PublicRecLast12M | MaxDelqEver | NumTotalTrades | NumTradesOpeninLast12M | PercentInstallTrades | MSinceMostRecentInqexcl7days | NumInqLast6M | NumInqLast6Mexcl7days | NetFractionRevolvingBurden | NetFractionInstallBurden | NumRevolvingTradesWBalance | NumInstallTradesWBalance| NumBank2NatlTradesWHighUtilization| PercentTradesWBalance|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
 $(PlutoRunner.@bind ExternalRiskEstimate NumberField(0:100000; default=59)) |  $(PlutoRunner.@bind MSinceOldestTradeOpen NumberField(0:100000; default=137)) |  $(PlutoRunner.@bind MSinceMostRecentTradeOpen NumberField(0:100000; default=11)) | $(PlutoRunner.@bind AverageMInFile NumberField(0:100000; default=78)) | $(PlutoRunner.@bind NumSatisfactoryTrades NumberField(0:100000; default=31)) | $(PlutoRunner.@bind NumTrades60Ever2DerogPubRec NumberField(0:100000; default=0)) | $(PlutoRunner.@bind NumTrades90Ever2DerogPubRec NumberField(0:100000; default=0)) | $(PlutoRunner.@bind PercentTradesNeverDelq NumberField(0:100000; default=91)) | $(PlutoRunner.@bind MSinceMostRecentDelq NumberField(0:100000; default=1)) | $(PlutoRunner.@bind MaxDelq2PublicRecLast12M NumberField(0:100000; default=4)) | $(PlutoRunner.@bind MaxDelqEver NumberField(0:100000; default=6)) | $(PlutoRunner.@bind NumTotalTrades NumberField(0:100000; default=32)) | $(PlutoRunner.@bind NumTradesOpeninLast12M NumberField(0:100000; default=1)) | $(PlutoRunner.@bind PercentInstallTrades NumberField(0:100000; default=47)) | $(PlutoRunner.@bind MSinceMostRecentInqexcl7days NumberField(0:100000; default=0))| $(PlutoRunner.@bind NumInqLast6M NumberField(0:100000; default=0))| $(PlutoRunner.@bind NumInqLast6Mexcl7days NumberField(0:100000; default=0))| $(PlutoRunner.@bind NetFractionRevolvingBurden NumberField(0:100000; default=62))| $(PlutoRunner.@bind NetFractionInstallBurden NumberField(0:100000; default=93))| $(PlutoRunner.@bind NumRevolvingTradesWBalance NumberField(0:100000; default=12))| $(PlutoRunner.@bind NumInstallTradesWBalance NumberField(0:100000; default=4))| $(PlutoRunner.@bind NumBank2NatlTradesWHighUtilization NumberField(0:100000; default=3))| $(PlutoRunner.@bind PercentTradesWBalance NumberField(0:100000; default=94))
"""

# ╔═╡ f1c13e1b-0861-4aa5-8459-bc0a7a72eaff
begin
	instance = deepcopy(X[1,:]);
	
	user_input = [ExternalRiskEstimate, MSinceOldestTradeOpen, MSinceMostRecentTradeOpen, AverageMInFile, NumSatisfactoryTrades, NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec, PercentTradesNeverDelq, MSinceMostRecentDelq, MaxDelq2PublicRecLast12M, MaxDelqEver, NumTotalTrades, NumTradesOpeninLast12M, PercentInstallTrades, MSinceMostRecentInqexcl7days, NumInqLast6M, NumInqLast6Mexcl7days, NetFractionRevolvingBurden, NetFractionInstallBurden, NumRevolvingTradesWBalance, NumInstallTradesWBalance, NumBank2NatlTradesWHighUtilization, PercentTradesWBalance]
	
	for (fidx, feature) in enumerate(propertynames(instance))
		instance[feature] = user_input[fidx]
	end
	pred = broadcast(MLJ.pdf, MLJ.predict(classifier, DataFrame(instance)),1)[1];
	#pred = classifier.predict([user_input])[1]
	goodness = (pred >= 0.5)
	
	if goodness
		md"Attention! The input instance is already classified as desired!"
	end
end

# ╔═╡ 4f53e07b-b7a2-401e-b5c0-d4817635db36
md""" 
**Step 4.** Run the explanation algorithm """

# ╔═╡ dc4a497c-7b19-11eb-2f9e-ed3c114f7761
explanations, _  = @time explain(instance, X, plaf_prog, classifier);

# ╔═╡ 7a6f48ad-8173-4ad5-bd4d-4a042fc49882
explanations[1:3,:]

# ╔═╡ b62dd892-5a2d-4a19-a5c4-4c8fc2221222
md"""
### Top Explanations

Show counterfactual: $(@PlutoRunner.bind cf_index NumberField(1:nrow(explanations); default=1))
"""

# ╔═╡ f2ea9cbc-ebf8-4287-bc2b-afa1a80ea160
if goodness 
	md"The provided instance does not require an explanation."
else 
	# Markdown.parse(GeCo.actions(explanations, instance; num_actions=actions, output="md"))
 	Markdown.parse(GeCo.actions(DataFrame(explanations[cf_index,:]), instance; num_actions = 2, output = "md"))
end

# ╔═╡ 801eb81e-a755-11eb-2d84-99fd8b51f552
orig_instance = X[6,:];

# ╔═╡ fa8320c0-1759-4f16-83ca-8329493b9dfa
begin
	groups = get_group(explanations, 10, orig_instance)
	options= ["$i"=>join(names(v),", ") for (i,v) in enumerate(groups)]

	md"""
	### Summary of Explanations
	
	Show Counterfactual Group: $(PlutoRunner.@bind group_index Select(options))"""
end

# ╔═╡ 997132af-d04c-4745-a42a-2563e07b5ca6
begin
	group_idx = parse(Int64, group_index)
	
	function generateGroupActions(groups, actions, orig_instance, group_id)
		group = groups[group_id]
		features = names(group)
		top = "**Changed Features:** $(join(features, ", "))\\\n"
		top *= "Number of counterfactuals: $(nrow(group))\\\n"
		top *= "Minimum Change:\\\n"

		cf = group[1,:]
		for feature in features
			top *= "-- $(feature) ： $(orig_instance[feature]) \$\\to\$ $(cf[feature])\\\n"
		end
		
		all = ""
		for (rid,cf) in enumerate(eachrow(groups[group_id]))
			all *= "**Counterfactual $rid**\\\n"
			for feature in features
				all *= "-- $(feature) ： $(orig_instance[feature]) \$\\to\$ $(cf[feature])\\\n"
			end
		end
		return top, all 
	end
	
	top, all = generateGroupActions(groups, actions, instance, group_idx)
	Markdown.parse(top)
end

# ╔═╡ 3245072c-e91e-418c-b8ce-63386333da8c
begin
	struct Foldable{C}
		title::String
		content::C
	end
	function Base.show(io, mime::MIME"text/html", fld::Foldable)
		write(io,"<details><summary>$(fld.title)</summary><p>")
		show(io, mime, fld.content)
		write(io,"</p></details>")
	end	

	Foldable(string("Show all suggested changes: "), Markdown.parse(all))
end 

# ╔═╡ 0bc25628-9d29-11eb-2d22-070e269ea036
begin
	function generate_explantions(X, classifier, p, K)
		
		# the result list -> one for each feature
		num_rows = nrow(X)
		num_features = ncol(X)
		
		weights = zeros(num_features)
		
		counts = zeros(num_features)
		pos_counts = zeros(num_features)
		neg_counts = zeros(num_features)
		
		cum_change = zeros(num_features)
		pos_cum_change = zeros(num_features)
		neg_cum_change = zeros(num_features)
		

		preds = MLJ.pdf.(MLJ.predict(classifier, X),1)
		instances = X[preds .<= 0.5, :][1:50, :]

		for cur_ins in eachrow(instances)
			explanations,  = explain(cur_ins, X, p, classifier)
			for i in 1:K
				explanation = explanations[i, :]
				indices = findall(explanation[:mod])
				for index in indices
					weights[index] += 1.0/length(indices)
					counts[index] += 1
					cum_change[index] += abs(cur_ins[index] - explanation[index])
					if (explanation[index] > cur_ins[index])
						pos_counts[index] += 1
						pos_cum_change[index] += 
							abs(cur_ins[index] - explanation[index])
					else
						neg_counts[index] += 1
						neg_cum_change[index] += 
							abs(cur_ins[index] - explanation[index])
					end

				end
			end
		end
		
		return instances, weights, counts, cum_change, pos_counts, neg_counts, 
			pos_cum_change, neg_cum_change
	end

	instances, weights, counts, cum_change, pos_counts, neg_counts, pos_cum_change, neg_cum_change = generate_explantions(X, classifier, p, 5)
	
	 
	md""" ## Explanations for multiple instances
	"""
end

# ╔═╡ 1113b964-9d29-11eb-19db-d1c5cb43d340
instances

# ╔═╡ a5043111-1600-4d6d-ac55-185d124abca9
Plots.bar(1:length(counts), counts,
	xticks=(1:length(counts), names(instance)),
	xrotation = -60,
	framestyle = :box,
	ylabel="Frequency",
	title="Total Count of Changes for Top 5 Explanations", 
	label="",
	)

# ╔═╡ 1954ed36-7393-4ff6-843a-17512f79a92b
groupedbar([pos_counts neg_counts],
	xticks=(1:length(counts), names(instance)),
	xrotation = -60,
	framestyle = :box,
	ylabel="Frequency",
	label=["Positive Change" "Negative Change"],
	title="Total Count of Change Directions for Top 5 Explanations", 
	)


# ╔═╡ 8f29bd6c-9d29-11eb-0e06-d9b5178cbe33
let 
	ave_change = [
		(counts[i] == 0) ? 0 : cum_change[i]/counts[i] 
		for i in 1:length(instance)
	]

	Plots.bar(1:length(ave_change), ave_change,
	xticks=(1:length(ave_change), names(instance)),
	xrotation = -60,
	framestyle = :box,
	ylabel="change value",
	title="Average Absolute Change", 
	label="",
	)
end

# ╔═╡ bdb171de-a756-11eb-2bee-bf3cfdbbf016
space = get_space(X);

# ╔═╡ e99670d2-a1a8-11eb-3cbb-49366adb2667
let 
	namesX = names(instance)
	ranges = space["ranges"]
	ave_change = [
		(counts[i] == 0) ? 0 : (cum_change[i]/counts[i]) / ranges[namesX[i]]
		for i in 1:length(namesX)
	]
	Plots.bar(1:length(ave_change), ave_change,
		xticks=(1:length(ave_change), names(instance)),
		xrotation = -60,
		framestyle = :box,
		ylabel="l1-distance",
		title = "Average Change w.r.t. Feature Range",
		label=""
		)

end

# ╔═╡ 95028f46-a1a9-11eb-276c-313e5d8bb515
# pos_counts, neg_counts, pos_cum_change, neg_cum_change
let 
	namesX = names(instance)
	pos_changes = []
	neg_changes = []

	ranges = space["ranges"]
	
	for i in 1:length(instance)
		if pos_counts[i] == 0
			append!(pos_changes, 0)
		else
			append!(pos_changes, 
				(pos_cum_change[i]/pos_counts[i])/ranges[namesX[i]])
		end
		if neg_counts[i] == 0
			append!(neg_changes, 0)
		else
			append!(neg_changes, 
				(neg_cum_change[i]/neg_counts[i])/ranges[namesX[i]])
		end
	end
	
	groupedbar([pos_changes neg_changes],
		xticks=(1:length(counts), names(instance)),
		xrotation = -60,
		framestyle = :box,
		ylabel="l1-distance",
		key=:outertopright,
		title="Average Change with Direction w.r.t. Feature Range",
		label=["Positive Change" "Negative Change"],
		)
end

# ╔═╡ 479acc12-915c-11eb-1a4b-f588eb756b0d
begin 
	feature_dict = Dict(feature => f_index 
		for (f_index, feature) in 	enumerate(names(orig_instance)))
end;

# ╔═╡ b342661f-3acc-4619-9593-fdd0f1014858
let 
	count = 1
	row_index = 1
	group = groups[group_idx]
	features = names(group)
	# xs = Array{String}(undef, nrow(group)*length(features))
	
	mins = space["mins"]
	maxs = space["maxs"]
	ranges = space["ranges"]
	
	ys = []

	for cf in eachrow(group)
		for feature in features
			append!(ys, (cf[feature]-mins[feature])/ranges[feature])
		end
	end
	
	ori_xs = Array{String}(undef,length(features))
	ori_ys = []
	for feature in features
		append!(ori_ys, 
			(user_input[feature_dict[feature]]-mins[feature]) / 
			ranges[feature])
		row_index += 1
	end
	
	gr()
		
	p = Plots.scatter(
		repeat(features,nrow(group)), ys,
		ylims=(0,1),
		ylabel="Normalized Range",
		label = "")
	
	scatter!(features, ori_ys, label = "")
	Foldable(string("Show plot: "), p)
end

# ╔═╡ Cell order:
# ╟─baadbc4c-7ba3-11eb-3cbb-a32fc7755748
# ╟─14adad40-6af7-4389-ae3a-b0d2a1462424
# ╟─1b2d0f6b-aab0-430f-9d92-67c6e585bc73
# ╠═3ebfd05d-8fac-4d4f-af0e-ba5d702014fa
# ╟─4a44f2a8-a75a-11eb-2d89-3fda270ecac3
# ╠═4e931614-a75a-11eb-3cbf-eff5cda26330
# ╟─08034110-b5a7-44c6-8a71-725998b43f60
# ╟─2a1fb272-4a2d-45c6-ba58-794fe3012c15
# ╟─f1c13e1b-0861-4aa5-8459-bc0a7a72eaff
# ╟─4f53e07b-b7a2-401e-b5c0-d4817635db36
# ╠═dc4a497c-7b19-11eb-2f9e-ed3c114f7761
# ╠═7a6f48ad-8173-4ad5-bd4d-4a042fc49882
# ╟─b62dd892-5a2d-4a19-a5c4-4c8fc2221222
# ╟─f2ea9cbc-ebf8-4287-bc2b-afa1a80ea160
# ╟─801eb81e-a755-11eb-2d84-99fd8b51f552
# ╟─fa8320c0-1759-4f16-83ca-8329493b9dfa
# ╟─997132af-d04c-4745-a42a-2563e07b5ca6
# ╟─3245072c-e91e-418c-b8ce-63386333da8c
# ╟─b342661f-3acc-4619-9593-fdd0f1014858
# ╟─0bc25628-9d29-11eb-2d22-070e269ea036
# ╠═1113b964-9d29-11eb-19db-d1c5cb43d340
# ╟─a5043111-1600-4d6d-ac55-185d124abca9
# ╟─1954ed36-7393-4ff6-843a-17512f79a92b
# ╟─8f29bd6c-9d29-11eb-0e06-d9b5178cbe33
# ╟─e99670d2-a1a8-11eb-3cbb-49366adb2667
# ╟─95028f46-a1a9-11eb-276c-313e5d8bb515
# ╟─bdb171de-a756-11eb-2bee-bf3cfdbbf016
# ╟─479acc12-915c-11eb-1a4b-f588eb756b0d
