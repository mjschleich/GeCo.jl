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

# ╔═╡ 543c16e2-7b18-11eb-3bba-fbd8cf708ca5
begin
	using Pkg; Pkg.activate(".")
	using GeCo
	include("scripts/credit/credit_setup_MACE.jl");
end;

# ╔═╡ a7036948-7b1d-11eb-1639-5d6b60931f40
begin 
	using PlutoUI
	using DataFrames
end

# ╔═╡ f491c536-7ba3-11eb-0c50-cfd7718f63e1
include("scripts/credit/credit_constraints_MACE.jl")

# ╔═╡ baadbc4c-7ba3-11eb-3cbb-a32fc7755748
md""" # The credit interactive setting experiment with user defined plaf constrains and any given orig_instance from dataset"""

# ╔═╡ 5981231a-7ba2-11eb-345d-d3a4a352a66e
begin
	p_user = initPLAF()
	
	@PLAF(p_user, :cf.isMale .== :x.isMale)
	@PLAF(p_user, :cf.isMarried .== :x.isMarried)
	
	@PLAF(p_user, :cf.AgeGroup .>= :x.AgeGroup)
	@PLAF(p_user, :cf.EducationLevel .>= :x.EducationLevel)
	@PLAF(p_user, :cf.HasHistoryOfOverduePayments .>= :x.HasHistoryOfOverduePayments)
	@PLAF(p_user, :cf.TotalOverdueCounts .>= :x.TotalOverdueCounts)
	@PLAF(p_user, :cf.TotalMonthsOverdue .>= :x.TotalMonthsOverdue)
	
	# If Education Level increases and Age is not adult, then move to next AgeGroup
	@PLAF(p_user, if cf.EducationLevel .> x.EducationLevel + 1 && x.AgeGroup .< 2; cf.AgeGroup == 2 end )
	
	# If MonthsWithLowSpendingOverLast6Months increases then MonthsWithHighSpendingOverLast6Months should decrease
	@PLAF(p_user, if cf.MonthsWithLowSpendingOverLast6Months .> x.MonthsWithLowSpendingOverLast6Months
	        cf.MonthsWithHighSpendingOverLast6Months .< x.MonthsWithHighSpendingOverLast6Months
	    end)
	
end

# ╔═╡ 956e796c-7b19-11eb-398a-0747c56c7fd2
orig_instance = X[5,:];

# ╔═╡ dc4a497c-7b19-11eb-2f9e-ed3c114f7761
explanation,  = @time explain(orig_instance, X,p_user, classifier);

# ╔═╡ c84900c4-9d28-11eb-11dc-7bdfed0e91af
md"""
* K(the number of counterfactuals show in)： $(@bind K_PLAF NumberField(1:100000; default=100))
"""

# ╔═╡ 0bc25628-9d29-11eb-2d22-070e269ea036
md""" ## plots for multiple instances explained
"""

# ╔═╡ 1113b964-9d29-11eb-19db-d1c5cb43d340
instances = X[1:100, :];

# ╔═╡ 2067e1ce-9d29-11eb-06a2-d7b0a7503416
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

# ╔═╡ 2d3be1de-9d29-11eb-3366-6f61831684f3
weights, counts, cum_change, pos_counts, neg_counts, pos_cum_change, neg_cum_change = generate_explantions(instances, X, classifier, p_user, 5);

# ╔═╡ 2f256966-9d29-11eb-2e63-8d6940fe66e1
import PyPlot

# ╔═╡ 7c44473a-9d29-11eb-0517-fb0dcaa4fa7f
PyPlot.svg(true);

# ╔═╡ 9841f2fe-7b2b-11eb-3e2d-f315722ca81d

md""" # The credit interactive setting experiment with user input values
"""

# ╔═╡ 120b4a66-7b1f-11eb-3b6a-ad33435c800a
md"""
* isMale： $(@bind isMale NumberField(1:100000; default=1))
* isMarried： $(@bind isMarried NumberField(1:100000; default=1))
* AgeGroup： $(@bind AgeGroup NumberField(1:100000; default=3))
* EducationLevel： $(@bind EducationLevel NumberField(1:100000; default=3))
* MaxBillAmountOverLast6Months： $(@bind MaxBillAmountOverLast6Months NumberField(1:100000; default=4650.0))
* MaxPaymentAmountOverLast6Months： $(@bind MaxPaymentAmountOverLast6Months NumberField(1:100000; default=450.0))
* MonthsWithZeroBalanceOverLast6Months： $(@bind MonthsWithZeroBalanceOverLast6Months NumberField(1:100000; default=0.0))    
* MonthsWithLowSpendingOverLast6Months： $(@bind MonthsWithLowSpendingOverLast6Months NumberField(1:100000; default=0.0))
* MonthsWithHighSpendingOverLast6Months： $(@bind MonthsWithHighSpendingOverLast6Months NumberField(1:100000; default=0.0))
* MostRecentBillAmount： $(@bind MostRecentBillAmount NumberField(1:100000; default=4140.0))
* MostRecentPaymentAmount： $(@bind MostRecentPaymentAmount NumberField(1:100000; default=200.0))
* TotalOverdueCounts： $(@bind TotalOverdueCounts NumberField(1:100000; default=1.0))
* TotalMonthsOverdue： $(@bind TotalMonthsOverdue NumberField(1:100000; default=17.0))
* HasHistoryOfOverduePayments：$(@bind HasHistoryOfOverduePayments NumberField(1:100000; default=1.0))
"""

# ╔═╡ dccd3e6a-7b30-11eb-0064-bbbe89fd7b20
begin
	instance = deepcopy(orig_instance);
	
	user_input = [isMale, isMarried, AgeGroup, EducationLevel, MaxBillAmountOverLast6Months, MaxPaymentAmountOverLast6Months, MonthsWithZeroBalanceOverLast6Months, MonthsWithLowSpendingOverLast6Months, MonthsWithHighSpendingOverLast6Months, MostRecentBillAmount, MostRecentPaymentAmount, TotalOverdueCounts, TotalMonthsOverdue, HasHistoryOfOverduePayments];

	nothing
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

# ╔═╡ cc7fbb06-1dda-4e40-9813-db6c79226583
for (fidx, feature) in enumerate(propertynames(instance))
	instance[feature] = user_input[fidx]
end

# ╔═╡ 55b3beaa-7b2f-11eb-0518-a5667d4747b7
user_explanations, goodness = explain(user_input, X, p, classifier);

# ╔═╡ 930d1f04-4734-4da5-ad84-52413e2198ac
goodness

# ╔═╡ 3ba087f6-9168-11eb-27b7-93a70a2104e9
if (goodness)
	md""" Attention! The input instance is already classified as desired!
	"""
end

# ╔═╡ 7cd85bbe-9130-11eb-3d5e-07187c2dcfcf
macro seeprints(input)
	expr = input
	quote
		stdout_bk = stdout
		rd, wr = redirect_stdout()
		$expr
		redirect_stdout(stdout_bk)
		close(wr)
		read(rd, String) |> Text
	end
end;

# ╔═╡ 3709c856-9172-11eb-0c8c-dd693faaac37
md"""### None Interactive Display (keep for possible later use)
"""

# ╔═╡ a1d5dfc9-b9bf-4e61-9124-ed1e9311c1a7
Markdown.parse(GeCo.actions(user_explanations, instance; num_actions=3))

# ╔═╡ bd5fa046-9164-11eb-29be-f322fc51efbb
md"""### TOP ACTIONS DISPLAY
"""

# ╔═╡ c9a8e04e-9162-11eb-3775-951d2f9e3933
md"""
Number of actions to display: $(@bind actions NumberField(1:100000; default=5))
"""

# ╔═╡ 5f7dacec-915b-11eb-25d3-75cbcf652e5e
function generate_top_action(goodness, user_explanations, actions, user_input)
	out = ""
	if  (!goodness)
		sort!(user_explanations, :score);
		for idx in 1:min(actions, nrow(user_explanations))
			cf = user_explanations[idx,:]
			out *= "\\\n**COUNTERFACTUAL $(idx)** \\\n Desired Outcome:$(cf.outc)\\\n   Score: $(cf.score)\\\n"
			for (f_index, feature) in enumerate(String.(names(orig_instance)))
				delta = cf[feature] - user_input[f_index]
				if delta != 0
					out *= "-- $feature : $(user_input[f_index]) \$\\to\$ $(cf[feature])\\\n"
				end
			end
		end
	end
	return out
end

# ╔═╡ 9587a30e-9d1c-11eb-3173-93df5aad5224
Markdown.parse(generate_top_action(goodness, user_explanations, actions, user_input))

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


# ╔═╡ 71bc086e-9164-11eb-255f-df25d4fc05a1
md"""### RESULT GROUP DISPLAY
"""

# ╔═╡ df6b33a8-9161-11eb-0e65-83fcc46a5f76
md"""
* K(the number of counterfactuals show in)： $(@bind K NumberField(1:100000; default=100))
"""

# ╔═╡ 0a34255c-9d38-11eb-32e0-b78e5703b16b
with_terminal() do 
	if  (!goodness)
		groups = get_group(user_explanations, K, user_input)
		for index in 0:length(groups)-1
			group = groups[index]
			features = String.(names(group))
			println("\n\nCOUNTERFACTUAL GROUP: $(features)")
			for r_index in 1:nrow(group)
				cf = group[r_index,:]
				println("\n--- COUNTERFACTUAL $(r_index)")
				for feature in features
					println("\t", feature, " : \t", user_input[feature_dict[feature]], " => ", cf[feature])
				end
			end
		end
	end
end

# ╔═╡ 9a2c2dc0-9140-11eb-3b17-95341317c7c6
function generate_group_action(goodness, user_explanations, actions, user_input, K)
	out = ""
	if  (!goodness)
		groups = get_group(user_explanations, K, user_input)
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
Markdown.parse(generate_group_action(false, explanation, actions, orig_instance, K_PLAF))

# ╔═╡ d6f3d9d4-9d1f-11eb-39dd-5bf9a91a6156
Markdown.parse(generate_group_action(goodness, user_explanations, actions, user_input, K))

# ╔═╡ c6c96fec-9d2e-11eb-1050-015257bb0640
let 
	if  (!goodness)
		groups = get_group(user_explanations, K, user_input)
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


# ╔═╡ 6d097d4a-9d37-11eb-12a3-6b8894ab3724
feature_text

# ╔═╡ Cell order:
# ╠═543c16e2-7b18-11eb-3bba-fbd8cf708ca5
# ╟─baadbc4c-7ba3-11eb-3cbb-a32fc7755748
# ╠═5981231a-7ba2-11eb-345d-d3a4a352a66e
# ╟─956e796c-7b19-11eb-398a-0747c56c7fd2
# ╠═dc4a497c-7b19-11eb-2f9e-ed3c114f7761
# ╠═c84900c4-9d28-11eb-11dc-7bdfed0e91af
# ╠═d5beb41a-9d28-11eb-2f1f-69bcc72b2005
# ╠═0bc25628-9d29-11eb-2d22-070e269ea036
# ╠═1113b964-9d29-11eb-19db-d1c5cb43d340
# ╠═2067e1ce-9d29-11eb-06a2-d7b0a7503416
# ╠═2d3be1de-9d29-11eb-3366-6f61831684f3
# ╠═2f256966-9d29-11eb-2e63-8d6940fe66e1
# ╠═7c44473a-9d29-11eb-0517-fb0dcaa4fa7f
# ╟─7fdfb708-9d29-11eb-3f7f-1372c5dabf2a
# ╟─30c1e278-9d2d-11eb-2580-bfbb921fe84a
# ╟─8f29bd6c-9d29-11eb-0e06-d9b5178cbe33
# ╠═80f49634-9d2b-11eb-359c-032422eaa82a
# ╠═ec374b8e-9d36-11eb-2060-7b6de3699a32
# ╠═1ef1a4ca-9d37-11eb-3b9f-7d5e93a9bef0
# ╠═9841f2fe-7b2b-11eb-3e2d-f315722ca81d
# ╠═a7036948-7b1d-11eb-1639-5d6b60931f40
# ╟─f491c536-7ba3-11eb-0c50-cfd7718f63e1
# ╠═dccd3e6a-7b30-11eb-0064-bbbe89fd7b20
# ╟─120b4a66-7b1f-11eb-3b6a-ad33435c800a
# ╟─cc7fbb06-1dda-4e40-9813-db6c79226583
# ╟─55b3beaa-7b2f-11eb-0518-a5667d4747b7
# ╟─930d1f04-4734-4da5-ad84-52413e2198ac
# ╟─3ba087f6-9168-11eb-27b7-93a70a2104e9
# ╟─7cd85bbe-9130-11eb-3d5e-07187c2dcfcf
# ╠═3709c856-9172-11eb-0c8c-dd693faaac37
# ╠═a1d5dfc9-b9bf-4e61-9124-ed1e9311c1a7
# ╟─bd5fa046-9164-11eb-29be-f322fc51efbb
# ╟─c9a8e04e-9162-11eb-3775-951d2f9e3933
# ╟─5f7dacec-915b-11eb-25d3-75cbcf652e5e
# ╟─9587a30e-9d1c-11eb-3173-93df5aad5224
# ╠═479acc12-915c-11eb-1a4b-f588eb756b0d
# ╠═075b9028-9161-11eb-0abb-d7f56278d55b
# ╠═a70ab516-913b-11eb-2f4c-5b3e31e2d669
# ╟─71bc086e-9164-11eb-255f-df25d4fc05a1
# ╠═df6b33a8-9161-11eb-0e65-83fcc46a5f76
# ╟─0a34255c-9d38-11eb-32e0-b78e5703b16b
# ╟─9a2c2dc0-9140-11eb-3b17-95341317c7c6
# ╟─d6f3d9d4-9d1f-11eb-39dd-5bf9a91a6156
# ╠═c6c96fec-9d2e-11eb-1050-015257bb0640
# ╠═6d097d4a-9d37-11eb-12a3-6b8894ab3724
