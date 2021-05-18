# we want to sample a new value based on the bayesian network for each entities
function causal_mutation!(population::DataFrame, network::BayesNet, feasible_space::FeasibleSpace; max_num_samples::Int64 = 5)
    # features = propertynames(DataFrame)
    # estcfs = population.estcf::BitVector
    groups::Vector{FeatureGroup} = feasible_space.groups
    sample_space::Vector{DataFrame} = feasible_space.feasibleSpace

    row = 1
    rows = nrow(population)
    while row <= rows
        instance_row = population[row,1:length(groups)]
        row_dic = Dict(propertynames(instance_row) .=> values(instance_row));
        
        
        entity = population[row,:]
        modified_features::BitVector = entity.mod::BitVector

        # The three lines below are to avoid deepcopies and pushing to DataFrames
        num_rows = length(groups) * max_num_samples
        mutatedInstances = DataFrame(entity)
        repeat!(mutatedInstances, num_rows)
        for i=1:num_rows
            mutatedInstances.mod[i] = BitArray{1}(modified_features)
            mutatedInstances.estcf[i] = false
        end

        # This BitVector is used to determine which mutations are valid
        validInstances = falses(num_rows)

        num_mutated_rows = 0
        # for feature in features
        for (index,group)  in enumerate(groups)
            feature = group.names[1]

            # # we will use this later to valid the samples
            df = sample_space[index]
            isempty(df) && continue;
            # create the evidence:
            evidence = copy(row_dic)
            delete!(evidence, feature)
            initial_sample = Assignment(row_dic)
            evidence_assignment = Assignment(evidence)
            # println(index)
            # println(initial_sample)
            # println(evidence_assignment)
            # create the sampler 
            gsampler = GibbsSampler(evidence_assignment,initial_sample=initial_sample)

            sampled_rows = rand(network, gsampler, max_num_samples)
            for s in 1:max_num_samples
                if (!isempty(df)) && sampled_rows[s, feature] <= maximum(df[!,feature]) && sampled_rows[s, feature] >= minimum(df[!,feature])
                    mutatedInstances[num_mutated_rows+s, feature] = sampled_rows[s, feature]
                
                    mutatedInstances[num_mutated_rows+s, :mod] .|= group.indexes

                    valid_action = actionCascade(mutatedInstances[num_mutated_rows+s, :], feasible_space.implications)
                    # !valid_action && println("We found an invalid action: ", valid_action)
                    validInstances[num_mutated_rows+s] = valid_action
                end
            end

            num_mutated_rows += max_num_samples
        end
        append!(population, mutatedInstances)
        row += 1
        # print(size(population))
    end
    unique!(population)
    # print(size(population))
end


# get the likelihood of the data 
function likelihood(df::DataFrame, num_features::Int64, network::BayesNet)
    results::Array{Float64,1} = zeros(Float64, nrow(df))
    for row in 1:nrow(df)
        results[row] = logpdf(network, DataFrame(df[row, 1:num_features]))
        # results[row] = pdf(network, df[row, 1:num_features])
    end
    return results
end
