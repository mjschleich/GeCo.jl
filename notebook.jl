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

	include("notebook/geco_rf_credit_setup.jl");
	
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

# ╔═╡ 87b7c5de-9238-4051-abea-0499ba648ae0
md""" 
**Step 2.** Define PLAF constraints to ensure plausibility and feasibilty of the explanations. """

# ╔═╡ 5981231a-7ba2-11eb-345d-d3a4a352a66e
begin
	plaf_prog = PLAFProgram()
	
	@PLAF(plaf_prog, cf.isMale .== x.isMale)
	@PLAF(plaf_prog, cf.isMarried .== x.isMarried)
	
	@PLAF(plaf_prog, cf.AgeGroup .>= x.AgeGroup)
	@PLAF(plaf_prog, cf.EducationLevel .>= x.EducationLevel)
	@PLAF(plaf_prog, cf.HasHistoryOfOverduePayments .>=
		x.HasHistoryOfOverduePayments)
	# @PLAF(plaf_prog, cf.TotalOverdueCounts .>= x.TotalOverdueCounts)
	# @PLAF(plaf_prog, cf.TotalMonthsOverdue .>= x.TotalMonthsOverdue)
	
	# If Education Level increases and Age is not adult, then move to next AgeGroup
	@PLAF(plaf_prog, 
		if cf.EducationLevel .> x.EducationLevel + 1 && x.AgeGroup .< 2;
			cf.AgeGroup == 2 end )
	
	# If MonthsWithLowSpendingOverLast6Months increases then 	MonthsWithHighSpendingOverLast6Months should decrease

	
	@PLAF(plaf_prog, 
		if cf.TotalOverdueCounts .> 
			x.TotalOverdueCounts;
			cf.MaxBillAmountOverLast6Months .> x.MaxBillAmountOverLast6Months - x.TotalOverdueCounts*3 end)
end

# ╔═╡ 08034110-b5a7-44c6-8a71-725998b43f60
md""" 
**Step 3.** Define the instance to be explained """

# ╔═╡ 2a1fb272-4a2d-45c6-ba58-794fe3012c15
md"""
| isMale | isMarried | AgeGroup | EducationLevel | MaxBillAmountLast6M | MaxPaymentAmountLast6M | MonthsWithZeroBalanceLast6M | MonthsWithLowSpendingLast6M | MonthsWithHighSpendingLast6M | MostRecentBillAmount | MostRecentPaymentAmount | TotalOverdueCounts | TotalMonthsOverdue | HasHistoryOfOverduePayments | 
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
 $(@bind isMale NumberField(1:100000; default=1)) |  $(@bind isMarried NumberField(1:100000; default=1)) |  $(@bind AgeGroup NumberField(1:100000; default=3)) | $(@bind EducationLevel NumberField(1:100000; default=3)) | $(@bind MaxBillAmountOverLast6Months NumberField(1:100000; default=4650.0)) | $(@bind MaxPaymentAmountOverLast6Months NumberField(1:100000; default=450.0)) | $(@bind MonthsWithZeroBalanceOverLast6Months NumberField(1:100000; default=0.0)) | $(@bind MonthsWithLowSpendingOverLast6Months NumberField(1:100000; default=0.0)) | $(@bind MonthsWithHighSpendingOverLast6Months NumberField(1:100000; default=0.0)) | $(@bind MostRecentBillAmount NumberField(1:100000; default=4140.0)) | $(@bind MostRecentPaymentAmount NumberField(1:100000; default=200.0)) | $(@bind TotalOverdueCounts NumberField(1:100000; default=1.0)) | $(@bind TotalMonthsOverdue NumberField(1:100000; default=17.0)) | $(@bind HasHistoryOfOverduePayments NumberField(0:1; default=1))
"""

# ╔═╡ f1c13e1b-0861-4aa5-8459-bc0a7a72eaff
begin
	instance = deepcopy(orig_instance);
	
	user_input = [isMale, isMarried, AgeGroup, EducationLevel, MaxBillAmountOverLast6Months, MaxPaymentAmountOverLast6Months, MonthsWithZeroBalanceOverLast6Months, MonthsWithLowSpendingOverLast6Months, MonthsWithHighSpendingOverLast6Months, MostRecentBillAmount, MostRecentPaymentAmount, TotalOverdueCounts, TotalMonthsOverdue, HasHistoryOfOverduePayments];
	
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

Show counterfactual: $(@bind cf_index NumberField(1:nrow(explanations); default=1))
"""

# ╔═╡ f2ea9cbc-ebf8-4287-bc2b-afa1a80ea160
if goodness 
	md"The provided instance does not require an explanation."
else 
 	Markdown.parse(GeCo.actions(DataFrame(explanations[cf_index,:]), instance; num_actions = 2, output = "md"))
end

# ╔═╡ fa8320c0-1759-4f16-83ca-8329493b9dfa
begin
	groups = get_group(explanations, 10, orig_instance)
	options= ["$i"=>join(names(v),", ") for (i,v) in enumerate(groups)]

	md"""
	### Summary of Explanations
	
	Show Counterfactual Group: $(@bind group_index Select(options))"""
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
		instances = X[preds .<= 0.5, :][1:100, :]

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

	instances, weights, counts, cum_change, pos_counts, neg_counts, pos_cum_change, neg_cum_change = generate_explantions(X, classifier, plaf_prog, 5)
	
	 
	md""" ## Explanations for multiple instances
	"""
end

# ╔═╡ 1113b964-9d29-11eb-19db-d1c5cb43d340
instances

# ╔═╡ a5043111-1600-4d6d-ac55-185d124abca9
Plots.bar(1:length(counts), counts,
	xticks=(1:length(counts), names(instance)),
	xrotation = 30,
	framestyle = :box,
	ylabel="Frequency",
	label="",
	title="Total Count of Changes for Top 5 Explanations", 
	)

# ╔═╡ 1954ed36-7393-4ff6-843a-17512f79a92b

groupedbar([pos_counts neg_counts],
	xticks=(1:length(counts), names(instance)),
	xrotation = 30,
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
	xrotation = 30,
	framestyle = :box,
	ylabel="change value",
	title="Average Absolute Change",
	label="",
	)
end

# ╔═╡ ce01f106-a19f-11eb-12e3-cd894166023d
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
		xrotation = 30,
		framestyle = :box,
		ylabel="l1-distance",
		title="Average Change w.r.t. Feature Range",
		label=""
		)
end

# ╔═╡ 30ba680e-a75a-11eb-0f4e-c51eaabb6944
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
		xrotation = 30,
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
		for (f_index, feature) in 	enumerate(names(orig_instance)));
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
# ╠═1b2d0f6b-aab0-430f-9d92-67c6e585bc73
# ╠═3ebfd05d-8fac-4d4f-af0e-ba5d702014fa
# ╟─87b7c5de-9238-4051-abea-0499ba648ae0
# ╠═5981231a-7ba2-11eb-345d-d3a4a352a66e
# ╟─08034110-b5a7-44c6-8a71-725998b43f60
# ╟─2a1fb272-4a2d-45c6-ba58-794fe3012c15
# ╟─f1c13e1b-0861-4aa5-8459-bc0a7a72eaff
# ╟─4f53e07b-b7a2-401e-b5c0-d4817635db36
# ╠═dc4a497c-7b19-11eb-2f9e-ed3c114f7761
# ╠═7a6f48ad-8173-4ad5-bd4d-4a042fc49882
# ╟─b62dd892-5a2d-4a19-a5c4-4c8fc2221222
# ╟─f2ea9cbc-ebf8-4287-bc2b-afa1a80ea160
# ╟─fa8320c0-1759-4f16-83ca-8329493b9dfa
# ╟─997132af-d04c-4745-a42a-2563e07b5ca6
# ╟─3245072c-e91e-418c-b8ce-63386333da8c
# ╟─b342661f-3acc-4619-9593-fdd0f1014858
# ╠═0bc25628-9d29-11eb-2d22-070e269ea036
# ╠═1113b964-9d29-11eb-19db-d1c5cb43d340
# ╟─a5043111-1600-4d6d-ac55-185d124abca9
# ╟─1954ed36-7393-4ff6-843a-17512f79a92b
# ╟─8f29bd6c-9d29-11eb-0e06-d9b5178cbe33
# ╟─e99670d2-a1a8-11eb-3cbb-49366adb2667
# ╟─30ba680e-a75a-11eb-0f4e-c51eaabb6944
# ╟─ce01f106-a19f-11eb-12e3-cd894166023d
# ╟─479acc12-915c-11eb-1a4b-f588eb756b0d
