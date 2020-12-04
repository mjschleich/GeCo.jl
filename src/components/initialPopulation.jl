## Generates the initial population
function initialPopulation(orig_entity, features, groups, feasible_space; compress_data::Bool=false)

    max_num_samples = 20
    num_rows = length(groups) * max_num_samples

    initial_pop = DataFrame(orig_entity)

    if compress_data
        initial_pop = initializeManager(orig_entity; extended=true)
    else
        repeat!(initial_pop, num_rows)
        insertcols!(initial_pop,
            :score=>zeros(Float64, num_rows),
            :outc=>falses(num_rows),
            :estcf=>falses(num_rows),
            :mod=>BitArray[falses(length(features)) for _=1:num_rows]
            )
    end

    row_index = 0
    for (index, group) in enumerate(groups)
        df = feasible_space[index]

        # if the feasible space is empty, continue to next group
        isempty(df) && continue;

        num_samples = min(max_num_samples, nrow(df))
        sampled_rows = StatsBase.sample(1:nrow(df), StatsBase.FrequencyWeights(df.count), num_samples; replace=false, ordered=true)

        if compress_data
            for s in 1:num_samples
                mod_df = falses(size(df, 2))
                mod_pop = falses(length(features))
                row = sampled_rows[s]

                mod_pop[group.indexes] .= true

                for (idx,fname) in enumerate(group.names)
                    fidx = group.indexes[idx]

                    # Set the mod for this row
                    mod_df[idx] = (df[row, fname] != orig_entity[fname])
                    # mod_pop[fidx] = (df[row, fname] != orig_entity[fname])
                end

                #println("mod: $mod_df / $mod_pop df[row,mod]: (df)")
                # push!(initial_pop, mod_pop, (df[row, mod_df]...,score=0.0, outc=false, estcf=false))
                push!(initial_pop, mod_pop, (df[row, 1:end-NUM_EXTRA_FEASIBLE_SPACE_COL]...,score=0.0, outc=false, estcf=false))
            end
        else
            for (idx,fname) in enumerate(group.names)
                fidx = group.indexes[idx]
                for s in 1:num_samples
                    initial_pop[row_index+s, fname] = df[sampled_rows[s], fname]
                    initial_pop[row_index+s, :mod][fidx] = 1 # (df[sampled_rows[s], fname] != orig_entity[fname])
                end
            end
        end
        row_index += num_samples
    end

    if compress_data
        return initial_pop
    else
        return initial_pop[1:row_index, :]
    end
end