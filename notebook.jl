### A Pluto.jl notebook ###
# v0.14.2

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

# ╔═╡ 543c16e2-7b18-11eb-3bba-fbd8cf708ca5
begin
	using GeCo
	
	using PlutoUI, PyPlot
	PyPlot.svg(true);

	include("notebook/geco_demo_credit_setup.jl");
	
	md"""Setup completed"""
end

# ╔═╡ baadbc4c-7ba3-11eb-3cbb-a32fc7755748
md""" 
# Demonstration of GeCo
"""



# ╔═╡ 14adad40-6af7-4389-ae3a-b0d2a1462424
md"""
**Step 1.** Provide a pre-learned classifier and dataset
"""

# ╔═╡ 1b2d0f6b-aab0-430f-9d92-67c6e585bc73
classifier

# ╔═╡ 3ebfd05d-8fac-4d4f-af0e-ba5d702014fa
summary(X)

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
	
	pred = classifier.predict([user_input])[1]
	goodness = (pred == 1)
	
	println(pred)
	
	if goodness
		md"Attention! The input instance is already classified as desired!"
	end
end

# ╔═╡ 4f53e07b-b7a2-401e-b5c0-d4817635db36
md""" 
**Step 4.** Run the explanation algorithm """

# ╔═╡ dc4a497c-7b19-11eb-2f9e-ed3c114f7761
explanations,_  = @time explain(instance, X, plaf_prog, classifier);

# ╔═╡ 7a6f48ad-8173-4ad5-bd4d-4a042fc49882
explanations[1:3,:]

# ╔═╡ b62dd892-5a2d-4a19-a5c4-4c8fc2221222
md"""
### Top Counterfactual Explanations

Number of actions to display: $(@bind actions NumberField(1:100000; default=5))
"""

# ╔═╡ f2ea9cbc-ebf8-4287-bc2b-afa1a80ea160
if goodness 
	md"The provided instance does not require an explanation."
else 
	Markdown.parse(GeCo.actions(explanations, instance; num_actions=actions))
end

# ╔═╡ c84900c4-9d28-11eb-11dc-7bdfed0e91af
md"""
### Summary of Generated Explanations

Number of groups： $(@bind K_PLAF NumberField(1:100000; default=3))
"""

# ╔═╡ 0bc25628-9d29-11eb-2d22-070e269ea036
md""" ## Explanations for 100 instances
"""

# ╔═╡ 1113b964-9d29-11eb-19db-d1c5cb43d340
instances = X[1:200, :];

# ╔═╡ 2067e1ce-9d29-11eb-06a2-d7b0a7503416
begin 
	function generate_explantions(instance, X, classifier, p, K)
		# the result list -> one for each feature
		num_rows = nrow(instance)
		num_features = ncol(X)
		weights = zeros(num_features)
		counts = zeros(num_features)
		pos_counts = zeros(num_features)
		neg_counts = zeros(num_features)
		cum_change = zeros(num_features)
		pos_cum_change = zeros(num_features)
		neg_cum_change = zeros(num_features)
		orig_frame = DataFrame(instance)
		insertcols!(orig_frame,
				:score=>zeros(Float64, num_rows),
				:outc=>falses(num_rows),
				:estcf=>falses(num_rows),
				:mod=>BitVector[falses(num_features) for _=1:num_rows]
				)
		scores = GeCo.score(classifier, orig_frame, 1)
		for row_num in 1:num_rows
			if scores[row_num] < 0.5
				# print(instance[row_num,:])
				cur_ins = instance[row_num,:]
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
		end
		return weights, counts, cum_change, pos_counts, neg_counts, pos_cum_change, neg_cum_change
	end;

	weights, counts, cum_change, pos_counts, neg_counts, pos_cum_change, neg_cum_change = generate_explantions(instances, X, classifier, plaf_prog, 5);
	
end

# ╔═╡ 7fdfb708-9d29-11eb-3f7f-1372c5dabf2a
let 
	namesX = names(instance)
	fig = PyPlot.matplotlib.pyplot.figure(figsize=(15, 6), dpi=80)
	a = []
	text = ""
	for i in 1:length(namesX)
		text *= "$(i) : $(namesX[i]) \n"
		append!(a, i)
	end
	
	box_sty = Dict([("boxstyle","round"), ("facecolor","wheat"), ("alpha",0.5)])

    PyPlot.clf() 
    PyPlot.bar(a, counts) 
	# PyPlot.scatter(namesX, weights) 
	PyPlot.matplotlib.pyplot.xlabel("Feature")
	PyPlot.matplotlib.pyplot.ylabel("Counts")
	PyPlot.matplotlib.pyplot.title("Counts For Each Feature Changed")
	
	
	PyPlot.text(length(namesX)+2, 0, text, style="italic", fontsize=12, bbox = box_sty)
    figure=PyPlot.gcf()
end

# ╔═╡ 30c1e278-9d2d-11eb-2580-bfbb921fe84a
let 
	namesX = names(instance)
	fig = PyPlot.matplotlib.pyplot.figure(figsize=(15, 6), dpi=80)
	a_1 = []
	a_2 = []
	text = ""
	for i in 1:length(namesX)
		text *= "$(i) : $(namesX[i]) \n"
		append!(a_1, i-0.2)
		append!(a_2, i+0.2)
	end
	
	box_sty = Dict([("boxstyle","round"), ("facecolor","wheat"), ("alpha",0.5)])

    PyPlot.clf()
	width = 0.4
	PyPlot.bar(a_1, pos_counts, width, color="cyan") 
	PyPlot.bar(a_2, neg_counts, width, color="orange") 
	# PyPlot.scatter(namesX, weights) 
	PyPlot.matplotlib.pyplot.xlabel("Feature")
	PyPlot.matplotlib.pyplot.ylabel("Counts")
	PyPlot.matplotlib.pyplot.title("Counts For Each Feature Changed")
	PyPlot.legend(["Positive Change", "Negative Change"])
	
	PyPlot.text(length(namesX)+2, 0, text, style="italic", fontsize=12, bbox = box_sty)
    figure=PyPlot.gcf()
end

# ╔═╡ 8f29bd6c-9d29-11eb-0e06-d9b5178cbe33
let 
	namesX = names(instance)
	fig = PyPlot.matplotlib.pyplot.figure(figsize=(10, 6), dpi=80)

    PyPlot.clf() 
    # PyPlot.bar(namesX, weights) 
	ave_change = []
	a = []
	max = 0
	text = ""
	for i in 1:length(namesX)
		if counts[i] == 0
			append!(ave_change, 0)
		else
			val = cum_change[i]/counts[i]
			if val > max
				max = val
			end
			append!(ave_change, val)
		end
		text *= "$(i) : $(namesX[i]) \n"
		append!(a, i)
	end
	PyPlot.bar(a, ave_change) 
	PyPlot.matplotlib.pyplot.xlabel("Feature")
	PyPlot.matplotlib.pyplot.ylabel("Average Changed")
	PyPlot.matplotlib.pyplot.title("Average Changed For Each Feature")
	
	box_sty = Dict([("boxstyle","round"), ("facecolor","wheat"), ("alpha",0.5)])
	PyPlot.text(length(namesX)+2, 0, text, style="italic", fontsize=12, bbox = box_sty)
	figure=PyPlot.gcf()
end

# ╔═╡ 80f49634-9d2b-11eb-359c-032422eaa82a
# pos_counts, neg_counts, pos_cum_change, neg_cum_change
let 
	namesX = names(instance)
	fig = PyPlot.matplotlib.pyplot.figure(figsize=(10, 6), dpi=80)

    PyPlot.clf() 
    # PyPlot.bar(namesX, weights) 
	pos_changes = []
	neg_changes = []
	a_1 = []
	a_2 = []
	text = ""
	for i in 1:length(namesX)
		if pos_counts[i] == 0
			append!(pos_changes, 0)
		else
			append!(pos_changes, pos_cum_change[i]/pos_counts[i])
		end
		if neg_counts[i] == 0
			append!(neg_changes, 0)
		else
			append!(neg_changes, neg_cum_change[i]/neg_counts[i])
		end
		text *= "$(i) : $(namesX[i]) \n"
		append!(a_1, i-0.2)
		append!(a_2, i+0.2)
	end
	width = 0.4
	PyPlot.bar(a_1, pos_changes, width, color="cyan") 
	PyPlot.bar(a_2, neg_changes, width, color="orange") 
	PyPlot.matplotlib.pyplot.xlabel("Feature")
	PyPlot.matplotlib.pyplot.ylabel("Average Changed")
	PyPlot.matplotlib.pyplot.title("Average Changed For Each Feature")
	PyPlot.legend(["Positive Change", "Negative Change"])
	box_sty = Dict([("boxstyle","round"), ("facecolor","wheat"), ("alpha",0.5)])
	PyPlot.text(length(namesX)+2, 0, text, style="italic", fontsize=12, bbox = box_sty)
	figure=PyPlot.gcf()
end

# ╔═╡ ec374b8e-9d36-11eb-2060-7b6de3699a32
function get_feature_text()
	feature_text = ""
	namesX = names(instance)
	for i in 1:length(namesX)
		feature_text *= "$(i) : $(namesX[i]) \n"
	end
	return feature_text
end


# ╔═╡ 1ef1a4ca-9d37-11eb-3b9f-7d5e93a9bef0
feature_text = get_feature_text()

# ╔═╡ 9841f2fe-7b2b-11eb-3e2d-f315722ca81d

md""" # The credit interactive setting experiment with user input values
"""

# ╔═╡ 479acc12-915c-11eb-1a4b-f588eb756b0d
function feature_to_index()
	feature_dict = Dict()
	for (f_index, feature) in enumerate(String.(names(orig_instance)))
		feature_dict[feature] = f_index
	end
	return feature_dict
end;

# ╔═╡ 075b9028-9161-11eb-0abb-d7f56278d55b
feature_dict = feature_to_index();

# ╔═╡ a70ab516-913b-11eb-2f4c-5b3e31e2d669
function get_group(user_explanations, k, ori_instance)
	sort!(user_explanations, :score)
	user_explanations = user_explanations[1:k,:]
	dict::Dict{Int64, DataFrame} = Dict()
	count = 0
	group_explanations = groupby(user_explanations, :mod)
	for group in group_explanations
		explanations_g = group[:,1:14]
		
		
		# we should filter out the rows that is dominate by others
		row_num = 2
		while row_num <= nrow(explanations_g)
			# check whether this row is a trivial one based on each of the previous rows
			cur_instance = explanations_g[row_num,:]
			trivial = false
			for prev_row in 1:row_num-1
				prev_instance = explanations_g[prev_row,:]
				check = 0
				for feature_index in 1:14
					if group[1, :mod][feature_index] == true && 
						!(ori_instance[feature_index] < prev_instance[feature_index] 
						<= cur_instance[feature_index] ||  
						ori_instance[feature_index] > prev_instance[feature_index] >= 
						cur_instance[feature_index])
						break
					end
					check += 1
				end
				if check == 14
					trivial = true
				end
			end
			if trivial
				deleterows!(explanations_g, row_num)
				row_num -= 1
			end
			row_num += 1
		end
		explanations_g = explanations_g[:, filter(x -> x = true, group[1, :mod])]
		unique!(explanations_g)
		dict[count] = explanations_g
		count += 1
	end
	return dict
end;


# ╔═╡ 06dbd968-7453-4226-aaea-1580819f8ec3
function generate_group_action(goodness, explanations, actions, user_input, K)
	out = ""
	if  (!goodness)
		groups = get_group(explanations, K, user_input)
		for index in 0:length(groups)-1
			group = groups[index]
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
end

# ╔═╡ d5beb41a-9d28-11eb-2f1f-69bcc72b2005
Markdown.parse(
	generate_group_action(false, explanations, actions, orig_instance, K_PLAF)
)

# ╔═╡ 71bc086e-9164-11eb-255f-df25d4fc05a1
md"""### RESULT GROUP DISPLAY
"""

# ╔═╡ 676827ac-ec82-4e22-9171-92494c99e435
begin
	K = 10
end

# ╔═╡ c6c96fec-9d2e-11eb-1050-015257bb0640
let 
	if  (!goodness)
		groups = get_group(explanations, K, user_input)
		f, axs = PyPlot.matplotlib.pyplot.subplots(1, length(groups), figsize=(25, 6))
		for index in 0:length(groups)-1
			row_index = 1
			group = groups[index]
			features = String.(names(group))
			xs = Array{String}(undef, nrow(group)*length(features))
			ys = []
			
			for r_index in 1:nrow(group)
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
			axs[index+1].scatter(xs, ys)
			axs[index+1].scatter(ori_xs, ori_ys)
			PyPlot.legend(["cf", "original"])
		end
		box_sty = Dict([("boxstyle","round"), ("facecolor","wheat"), ("alpha",0.5)])
		PyPlot.text(length(names(groups[length(groups)-1]))-0.7, 0, feature_text, style="italic", fontsize=12, bbox = box_sty)
		figure=PyPlot.gcf()		
	end
end


# ╔═╡ Cell order:
# ╟─543c16e2-7b18-11eb-3bba-fbd8cf708ca5
# ╠═06dbd968-7453-4226-aaea-1580819f8ec3
# ╟─baadbc4c-7ba3-11eb-3cbb-a32fc7755748
# ╟─14adad40-6af7-4389-ae3a-b0d2a1462424
# ╠═1b2d0f6b-aab0-430f-9d92-67c6e585bc73
# ╠═3ebfd05d-8fac-4d4f-af0e-ba5d702014fa
# ╟─87b7c5de-9238-4051-abea-0499ba648ae0
# ╠═5981231a-7ba2-11eb-345d-d3a4a352a66e
# ╟─08034110-b5a7-44c6-8a71-725998b43f60
# ╟─2a1fb272-4a2d-45c6-ba58-794fe3012c15
# ╠═f1c13e1b-0861-4aa5-8459-bc0a7a72eaff
# ╟─4f53e07b-b7a2-401e-b5c0-d4817635db36
# ╠═dc4a497c-7b19-11eb-2f9e-ed3c114f7761
# ╠═7a6f48ad-8173-4ad5-bd4d-4a042fc49882
# ╟─b62dd892-5a2d-4a19-a5c4-4c8fc2221222
# ╟─f2ea9cbc-ebf8-4287-bc2b-afa1a80ea160
# ╟─c84900c4-9d28-11eb-11dc-7bdfed0e91af
# ╟─d5beb41a-9d28-11eb-2f1f-69bcc72b2005
# ╟─0bc25628-9d29-11eb-2d22-070e269ea036
# ╠═1113b964-9d29-11eb-19db-d1c5cb43d340
# ╟─2067e1ce-9d29-11eb-06a2-d7b0a7503416
# ╟─7fdfb708-9d29-11eb-3f7f-1372c5dabf2a
# ╟─30c1e278-9d2d-11eb-2580-bfbb921fe84a
# ╟─8f29bd6c-9d29-11eb-0e06-d9b5178cbe33
# ╟─80f49634-9d2b-11eb-359c-032422eaa82a
# ╟─ec374b8e-9d36-11eb-2060-7b6de3699a32
# ╟─1ef1a4ca-9d37-11eb-3b9f-7d5e93a9bef0
# ╟─9841f2fe-7b2b-11eb-3e2d-f315722ca81d
# ╠═479acc12-915c-11eb-1a4b-f588eb756b0d
# ╠═075b9028-9161-11eb-0abb-d7f56278d55b
# ╠═a70ab516-913b-11eb-2f4c-5b3e31e2d669
# ╟─71bc086e-9164-11eb-255f-df25d4fc05a1
# ╠═676827ac-ec82-4e22-9171-92494c99e435
# ╠═c6c96fec-9d2e-11eb-1050-015257bb0640
