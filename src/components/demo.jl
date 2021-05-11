function get_group(user_explanations, k, ori_instance)
    # sort!(user_explanations, :score)

    groups = Vector{DataFrame}()
    group_explanations = groupby(user_explanations, :mod)

    for (group_index,group) in enumerate(group_explanations)
        explanations_g = group[:,1:size(user_explanations)[2]-4]
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

function get_space(X)
	ranges = Dict(String(feature) => max(1.0, Float64(maximum(col)-minimum(col))) 
			for (feature, col) in pairs(eachcol(X)))
	maxs = Dict(String(feature) => Float64(maximum(col)) 
			for (feature, col) in pairs(eachcol(X)))
	mins = Dict(String(feature) => Float64(minimum(col)) 
			for (feature, col) in pairs(eachcol(X)))
    return Dict("ranges" => ranges, "maxs" => maxs,"mins" => mins)
end

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
end