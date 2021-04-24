### A Pluto.jl notebook ###
# v0.14.3

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
		if cf.MonthsWithLowSpendingOverLast6Months .> 
			x.MonthsWithLowSpendingOverLast6Months;
			cf.MonthsWithHighSpendingOverLast6Months .<
			x.MonthsWithHighSpendingOverLast6Months
	end)
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
	# Markdown.parse(GeCo.actions(explanations, instance; num_actions=actions, output="md"))
 	Markdown.parse(GeCo.actionsDemo(explanations, instance)[cf_index])
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
		instances = X[preds .>= 0.5, :][1:100, :]

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
instances[1:100,:]

# ╔═╡ a5043111-1600-4d6d-ac55-185d124abca9
Plots.bar(1:length(counts), counts,
	xticks=(1:length(counts), names(instance)),
	xrotation = 30,
	framestyle = :box,
	ylabel="Frequency",
	label="",
	)

# ╔═╡ 1954ed36-7393-4ff6-843a-17512f79a92b

groupedbar([pos_counts neg_counts],
	xticks=(1:length(counts), names(instance)),
	xrotation = 30,
	framestyle = :box,
	ylabel="Frequency",
	label=["Positive Change" "Negative Change"],
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
	ylabel="l1-distance",
	title="Average Absolute Change",
	label="",
	)
end

# ╔═╡ 80f49634-9d2b-11eb-359c-032422eaa82a
# pos_counts, neg_counts, pos_cum_change, neg_cum_change
let 
	pos_changes = [
		(pos_counts[i] == 0) ? 0 : pos_cum_change[i]/pos_counts[i] 
		for i in 1:length(instance)
	]
	neg_changes = [
		( neg_counts[i] == 0 ) ? 0 : neg_cum_change[i]/neg_counts[i]
		for i in 1:length(instance)
	]

	groupedbar([pos_counts neg_counts],
		xticks=(1:length(counts), names(instance)),
		xrotation = 30,
		framestyle = :box,
		ylabel="Frequency",
		# title="Average Changed For Each Feature",
		label=["Positive Change" "Negative Change"],
		)
end

# ╔═╡ 95490a61-0aa7-40f0-aa78-da9a2db11852
md"### Auxiliary Functions"

# ╔═╡ a70ab516-913b-11eb-2f4c-5b3e31e2d669
begin 
	function get_group(user_explanations, k, ori_instance)
		# sort!(user_explanations, :score)

		groups = Vector{DataFrame}()
		group_explanations = groupby(user_explanations, :mod)

		for (group_index,group) in enumerate(group_explanations)
			explanations_g = group[:,1:14]
			explanations_g = explanations_g[:, filter(x -> x = true, group[1, :mod])]

			# we should filter out the rows that is dominate by others
			row_num = 1

			keep_rows = trues(nrow(explanations_g))

			for (row_num, cur_instance) in enumerate(eachrow(explanations_g))
				# check if this row is dominated by previous rows and groups

				dominatedrow = false

				# check for each of the previous groups
				for prev_group in groups

					if !issubset(propertynames(prev_group), 
							propertynames(explanations_g))
						continue
					end

					for prev_group_row in eachrow(prev_group)
						isdominated = true
						for feature in propertynames(prev_group)
							if !(ori_instance[feature] < 
									prev_group_row[feature] <=
									cur_instance[feature] 
									||  
									ori_instance[feature] > 
									prev_group_row[feature] >= 
									cur_instance[feature])
								isdominated = false 
								break
							end
						end
						if isdominated
							dominatedrow = true
							break
						end
					end
					if dominatedrow
						break
					end
				end
				if dominatedrow
					keep_rows[row_num] = false 
					continue
				end

				# check for each of the previous rows
				for prev_row in 1:row_num-1
					prev_instance = explanations_g[prev_row,:]
					isdominated = true
					for feature in propertynames(explanations_g)
						if !(ori_instance[feature] <
								prev_instance[feature] <= 
								cur_instance[feature] 
							||  
								ori_instance[feature] > 
								prev_instance[feature] >= 
								cur_instance[feature]
							)
							isdominated = false
							break
						end
					end
					if isdominated
						dominatedrow = true
						break
					end
				end
				if dominatedrow
					keep_rows[row_num] = false 
				end
			end

			explanations_g = unique(explanations_g[keep_rows, :])
			if !isempty(explanations_g)
				push!(groups, explanations_g)
			end
		end
		return groups
	end
	
	md"get group"
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

# ╔═╡ ce01f106-a19f-11eb-12e3-cd894166023d
begin 
	function get_space()
	ranges = Dict(String(feature) => max(1.0, Float64(maximum(col)-minimum(col))) 
			for (feature, col) in pairs(eachcol(X)))
	maxs = Dict(String(feature) => Float64(maximum(col)) 
			for (feature, col) in pairs(eachcol(X)))
	mins = Dict(String(feature) => Float64(minimum(col)) 
			for (feature, col) in pairs(eachcol(X)))
		return Dict("ranges" => ranges, "maxs" => maxs,"mins" => mins)
	end
	space = get_space();
	
	md"feature space"
end

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
		xrotation = 30,
		framestyle = :box,
		ylabel="l1-distance",
		key=:outertopright,
		title="Average Change w.r.t. Feature Range",
		label=["Positive Change" "Negative Change"],
		)
end

# ╔═╡ 479acc12-915c-11eb-1a4b-f588eb756b0d
begin 
	feature_dict = Dict(feature => f_index 
		for (f_index, feature) in 	enumerate(names(orig_instance)));
		
	md" feature_dict "
end

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

# ╔═╡ 06dbd968-7453-4226-aaea-1580819f8ec3
begin 
	function generate_group_action(goodness, explanations, actions, user_input, K)
		out = ""
		if  (!goodness)
			groups = get_group(explanations, K, user_input)
			for group in groups
				features = String.(names(group))
				out *= "\\\n**COUNTERFACTUAL GROUP : $(features)**\\\n"
				for r_index in 1:nrow(group)
					cf = group[r_index,:]
					out *= " -- COUNTERFACTUAL $(r_index)\\\n"
					for feature in features
						out *= "$(feature) ： $(user_input[feature_dict[feature]]) \$\\to\$ $(cf[feature])\\\n"
					end
				end
			end
		end
		return out
	end;
	md"generate group action"
end

# ╔═╡ c8b1d8ee-8517-4e79-b1fa-429f659da38e
# begin
# 	lindex = 3; uindex=5
# 	md"""
# 	### Top Counterfactual Explanations
# 	Number of actions to display: $(@bind actions2 NumberField(1:100000; default=6))
# 	$(@bind actions3 NumberField(actions2:100000; default=5))
# 	"""
# end

# ╔═╡ 498293d7-58aa-4be1-90de-d0b5ce5c8b42
#= html"""<b style="font-family:'JuliaMono';">Counterfactual</b> 
<details close>
<summary style="font-family:JuliaMono;">Want to ruin the surprise?</summary>
....
</details>""" =#

# ╔═╡ 0a72330b-27e2-494e-bbfc-c0fc6cff4d94
md"### Previous Group Function"

# ╔═╡ c84900c4-9d28-11eb-11dc-7bdfed0e91af
md"""
### Summary of Generated Explanations

Number of groups： $(@bind K_PLAF NumberField(1:100000; default=3))
"""

# ╔═╡ d5beb41a-9d28-11eb-2f1f-69bcc72b2005
Markdown.parse(
	generate_group_action(false, explanations, actions, orig_instance, K_PLAF)
)

# ╔═╡ 4d0dcd7a-a1a1-11eb-305a-fb9789300d55
let 
	if  (!goodness)
		plot_ret = nothing
		count = 1
		groups = get_group(explanations, K_PLAF, instance)

		for group in groups #  index in 0:length(groups)-1
			row_index = 1
			features = String.(names(group))
			xs = Array{String}(undef, nrow(group)*length(features))
			ys = []
			
			for r_index in 1:DataFrames.nrow(group)
				cf = group[r_index,:]
				for feature in features
					xs[row_index] = string(feature_dict[feature])
					row_index += 1
					append!(ys, cf[feature])
				end
			end
			ori_xs = Array{String}(undef,length(features))
			ori_ys = []
			row_index = 1
			for feature in features
				ori_xs[row_index] = string(feature_dict[feature])
				append!(ori_ys, user_input[feature_dict[feature]])
				row_index += 1
			end
			
			p_cur = Plots.scatter(xs, ys, label = "cf")
			scatter!(ori_xs, ori_ys, label = "ori")
			if count == 1
				plot_ret = p_cur 
			elseif count == length(groups)
				plot_ret = Plots.plot(plot_ret, p_cur, 
					layout = Plots.grid(1, 2, widths=[1-1/count, 1/count]))
			else
				plot_ret = Plots.plot(plot_ret, p_cur, 
					layout = Plots.grid(1, 2, widths=[1-1/count, 1/count]))
			end
			count += 1
			
		end
		plot_ret
		
	end
end;

# ╔═╡ 71f98784-a1a4-11eb-2afd-b70324f6ade1
let 
	if  (!goodness)
		plot_ret = nothing
		count = 1
		groups = get_group(explanations, K_PLAF, instance)

		for group in groups # index in 0:length(groups)-1
			row_index = 1
			features = String.(names(group))
			xs = Array{String}(undef, nrow(group)*length(features))
			ys = []
			
			for r_index in 1:DataFrames.nrow(group)
				cf = group[r_index,:]
				for feature in features
					xs[row_index] = string(feature_dict[feature])
					row_index += 1
					append!(ys, (cf[feature]-space["mins"][feature]) / space["ranges"][feature])
				end
			end
			ori_xs = Array{String}(undef,length(features))
			ori_ys = []
			row_index = 1
			for feature in features
				ori_xs[row_index] = string(feature_dict[feature])
				append!(ori_ys, 
					(user_input[feature_dict[feature]]-space["mins"][feature]) / 
					space["ranges"][feature])
				row_index += 1
			end
			
			p_cur = Plots.scatter(xs, ys, label = "")
			scatter!(ori_xs, ori_ys, label = "")
			if count == 1
				plot_ret = p_cur 
			elseif count == length(groups)
				plot_ret = Plots.plot(plot_ret, p_cur, 
					layout = Plots.grid(1, 2, widths=[1-1/count, 1/count]))
			else
				plot_ret = Plots.plot(plot_ret, p_cur, 
					layout = Plots.grid(1, 2, widths=[1-1/count, 1/count]))
			end
			count += 1
			
		end
		plot_ret
	end
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
# ╟─0bc25628-9d29-11eb-2d22-070e269ea036
# ╠═1113b964-9d29-11eb-19db-d1c5cb43d340
# ╟─a5043111-1600-4d6d-ac55-185d124abca9
# ╟─1954ed36-7393-4ff6-843a-17512f79a92b
# ╟─8f29bd6c-9d29-11eb-0e06-d9b5178cbe33
# ╟─e99670d2-a1a8-11eb-3cbb-49366adb2667
# ╟─80f49634-9d2b-11eb-359c-032422eaa82a
# ╟─95028f46-a1a9-11eb-276c-313e5d8bb515
# ╟─95490a61-0aa7-40f0-aa78-da9a2db11852
# ╟─a70ab516-913b-11eb-2f4c-5b3e31e2d669
# ╟─ce01f106-a19f-11eb-12e3-cd894166023d
# ╟─06dbd968-7453-4226-aaea-1580819f8ec3
# ╟─479acc12-915c-11eb-1a4b-f588eb756b0d
# ╠═c8b1d8ee-8517-4e79-b1fa-429f659da38e
# ╠═498293d7-58aa-4be1-90de-d0b5ce5c8b42
# ╟─0a72330b-27e2-494e-bbfc-c0fc6cff4d94
# ╟─c84900c4-9d28-11eb-11dc-7bdfed0e91af
# ╟─d5beb41a-9d28-11eb-2f1f-69bcc72b2005
# ╟─4d0dcd7a-a1a1-11eb-305a-fb9789300d55
# ╟─71f98784-a1a4-11eb-2afd-b70324f6ade1
