### A Pluto.jl notebook ###
# v0.14.1

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
explanation,  = @time explain(orig_instance, X,p_user, classifier)

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
with_terminal() do 
	if  (!goodness)
		sort!(user_explanations, :score);
		for idx in 1:min(actions, nrow(user_explanations))
			cf = user_explanations[idx,:]
			println("\n------- COUNTERFACTUAL $idx\nDesired Outcome: $(cf.outc),\tScore: $(cf.score)")
			for (f_index, feature) in enumerate(String.(names(orig_instance)))
				delta = cf[feature] - user_input[f_index]
				if delta != 0
					println(feature, " : \t",user_input[f_index], " => ", cf[feature])
				end
			end
		end
	end
end

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
function get_group(user_explanations, k)
	sort!(user_explanations, :score)
	user_explanations = user_explanations[1:k,:]
	dict::Dict{Int64, DataFrame} = Dict()
	count = 0
	group_explanations = groupby(user_explanations, :mod)
	for group in group_explanations
		explanations_g = group[:,1:14]
		explanations_g = explanations_g[:, filter(x -> x = true, group[1, :mod])]
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

# ╔═╡ 563d910a-9139-11eb-1215-89384f244a32
@seeprints for idx in 1:min(K, nrows(user_explanations))
	cf = user_explanations[idx,:]
	println("\n------- COUNTERFACTUAL $idx\nDesired Outcome: $(cf.outc),\tScore: $(cf.score)")
	for (f_index, feature) in enumerate(String.(names(orig_instance)))
		delta = cf[feature] - user_input[f_index]
		if delta != 0
			println(feature, " : \t",user_input[f_index], " => ", cf[feature])
		end
	end
end


# ╔═╡ 9a2c2dc0-9140-11eb-3b17-95341317c7c6
with_terminal() do 
	if  (!goodness)
		groups = get_group(user_explanations, K)
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

# ╔═╡ Cell order:
# ╠═543c16e2-7b18-11eb-3bba-fbd8cf708ca5
# ╟─baadbc4c-7ba3-11eb-3cbb-a32fc7755748
# ╠═5981231a-7ba2-11eb-345d-d3a4a352a66e
# ╟─956e796c-7b19-11eb-398a-0747c56c7fd2
# ╠═dc4a497c-7b19-11eb-2f9e-ed3c114f7761
# ╠═9841f2fe-7b2b-11eb-3e2d-f315722ca81d
# ╟─a7036948-7b1d-11eb-1639-5d6b60931f40
# ╟─f491c536-7ba3-11eb-0c50-cfd7718f63e1
# ╟─dccd3e6a-7b30-11eb-0064-bbbe89fd7b20
# ╠═120b4a66-7b1f-11eb-3b6a-ad33435c800a
# ╟─cc7fbb06-1dda-4e40-9813-db6c79226583
# ╠═55b3beaa-7b2f-11eb-0518-a5667d4747b7
# ╠═930d1f04-4734-4da5-ad84-52413e2198ac
# ╟─3ba087f6-9168-11eb-27b7-93a70a2104e9
# ╟─7cd85bbe-9130-11eb-3d5e-07187c2dcfcf
# ╟─3709c856-9172-11eb-0c8c-dd693faaac37
# ╟─563d910a-9139-11eb-1215-89384f244a32
# ╠═a1d5dfc9-b9bf-4e61-9124-ed1e9311c1a7
# ╟─bd5fa046-9164-11eb-29be-f322fc51efbb
# ╠═c9a8e04e-9162-11eb-3775-951d2f9e3933
# ╠═5f7dacec-915b-11eb-25d3-75cbcf652e5e
# ╟─479acc12-915c-11eb-1a4b-f588eb756b0d
# ╟─075b9028-9161-11eb-0abb-d7f56278d55b
# ╠═a70ab516-913b-11eb-2f4c-5b3e31e2d669
# ╟─71bc086e-9164-11eb-255f-df25d4fc05a1
# ╟─df6b33a8-9161-11eb-0e65-83fcc46a5f76
# ╠═9a2c2dc0-9140-11eb-3b17-95341317c7c6
