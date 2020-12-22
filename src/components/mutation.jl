
####
## The mutation operator
####

function mutation!(
    population::DataFrame, feasible_space::FeasibleSpace; max_num_samples::Int64 = 5)

    groups::Vector{FeatureGroup} = feasible_space.groups
    sample_space::Vector{DataFrame} = feasible_space.feasibleSpace

    row = 1
    while row < nrow(population) && population[row, :estcf]
        entity = population[row,:]
        modified_features::BitVector = entity.mod

        # The three lines below are to avoid deepcopies and pushing to DataFrames
        num_rows = length(groups) * max_num_samples
        mutatedInstances = DataFrame(entity)
        repeat!(mutatedInstances, num_rows)
        for i=1:num_rows
            mutatedInstances.mod[i] = BitArray{1}(modified_features)
            mutatedInstances.estcf[i] = false
        end

        num_mutated_rows = 0
        for (index,group)  in enumerate(groups)
            df = sample_space[index]

            (isempty(df) || any(modified_features[group.indexes])) && continue;

            num_samples = min(max_num_samples, nrow(df))
            sampled_rows = StatsBase.sample(1:nrow(df), StatsBase.FrequencyWeights(df.count), num_samples; replace=false, ordered=true)

            for (findex,fname) in enumerate(group.names)
                # fidx = group.indexes[findex]
                for s in 1:num_samples
                    mutatedInstances[num_mutated_rows+s, fname] = df[sampled_rows[s], fname]
                end
            end

            for s in 1:num_samples
                mutatedInstances[num_mutated_rows+s, :mod] .|= group.indexes
            end

            num_mutated_rows += num_samples
        end
        # global num_generated += num_mutated_rows
        append!(population, mutatedInstances[1:num_mutated_rows, :])
        row += 1
    end
end

function mutation!(manager::DataManager, feasible_space::FeasibleSpace; max_num_samples::Int64 = 5)

    groups::Vector{FeatureGroup} = feasible_space.groups
    sample_space::Vector{DataFrame} = feasible_space.feasibleSpace

    keyset = collect(keys(manager))

    for mod in keyset

        population = manager.dict[mod]

        row = 1
        while row <= size(population, 1) && population[row, :estcf]
            entity_df = DataFrame(population[row, :])
            entity_df.estcf[1] = false
            repeat!(entity_df, max_num_samples)

            for (index,group)  in enumerate(groups)
                df = sample_space[index]

                (isempty(df) || any(mod[group.indexes])) && continue;

                num_samples = min(max_num_samples, nrow(df))
                sampled_rows = StatsBase.sample(1:nrow(df), StatsBase.FrequencyWeights(df.count), num_samples; replace=false, ordered=true)

                refined_modified_features = mod .| group.indexes

                # Would it be safe to use copycols=false here?
                mutatedInstances = hcat(
                    entity_df[1:num_samples,:],
                    df[sampled_rows, 1:end-NUM_EXTRA_FEASIBLE_SPACE_COL])
                # mutatedInstances[:,:mod] = (refined_modified_features for i=1:nrow(mutatedInstances))

                append!(manager, refined_modified_features, mutatedInstances)
            end

            row += 1
        end
    end
end
