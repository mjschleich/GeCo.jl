

function initDomains(path::String, data::DataFrame)::Vector{DataFrame}
    _, groups = initializeFeatures(path*"/data_info.json", data)
    return initDomains(groups, data)
end

function initDomains(groups::Vector{FeatureGroup}, data::DataFrame)::Vector{DataFrame}

    feasible_space = Vector{DataFrame}(undef, length(groups))
    for (gidx, group) in enumerate(groups)
        feat_names = group.names
        feat_sym = Symbol.(feat_names)
        if length(feat_sym) > 1000
            feasible_space[gidx] = data[!, feat_sym]
            feasible_space[gidx].count = ones(Int64, nrow(data))
        else
            feasible_space[gidx] = combine(groupby( data[!, feat_sym], feat_sym), nrow => :count)
        end
    end
    return feasible_space
end

function feasibleSpace(orig_instance::DataFrameRow, groups::Vector{FeatureGroup}, domains::Vector{DataFrame})::Vector{DataFrame}
    # We create a dictionary of constraints for each feature
    # A constraint is a function
    feasibility_constraints = Dict{String,Function}()
    feasible_space = Vector{DataFrame}(undef, length(groups))

    @assert length(domains) == length(groups)

    # This should come from the constraint parser
    # actionable_features = []

    for (gidx, group) in enumerate(groups)

        domain = domains[gidx]

        feasible_rows = trues(nrow(domain))
        identical_rows = trues(nrow(domain))

        for feature in group.features

            orig_val::Float64 = orig_instance[feature.name]
            col::Vector{Float64} = domain[!, feature.name]

            identical_rows .&= (col .== orig_val)

            if !feature.actionable
                feasible_rows .&= (col .== orig_val)
            elseif feature.constraint == INCREASING
                feasible_rows .&= (col .> orig_val)
            elseif feature.constraint == DECREASING
                feasible_rows .&= (col .< orig_val)
            end
        end

        # Remove the rows that have identical values - we don't want to sample the original instance
        feasible_rows .&= .!identical_rows

        feasible_space[gidx] = domains[gidx][feasible_rows, :]
        feasible_space[gidx].distance = distanceFeatureGroup(feasible_space[gidx], orig_instance, group)
    end

    return feasible_space
end


function feasibleSpace(orig_instance::DataFrameRow, groups::Vector{FeatureGroup}, data::DataFrame)::Vector{DataFrame}
    # We create a dictionary of constraints for each feature
    # A constraint is a function
    feasibility_constraints = Dict{String,Function}()
    feasible_space = Vector{DataFrame}(undef, length(groups))

    # This should come from the constraint parser
    actionable_features = []

    feasible_rows = trues(size(data,1))
    identical_rows = trues(size(data,1))

    for (gidx, group) in enumerate(groups)

        fill!(feasible_rows, true)
        fill!(identical_rows, true)

        feat_names = String[]
        for feature in group.features
            identical_rows .&= (data[:, feature.name] .== orig_instance[feature.name])

            if !feature.actionable
                feasible_rows .&= (data[:, feature.name] .== orig_instance[feature.name])
            elseif feature.constraint == INCREASING
                feasible_rows .&= (data[:, feature.name] .> orig_instance[feature.name])
            elseif feature.constraint == DECREASING
                feasible_rows .&= (data[:, feature.name] .< orig_instance[feature.name])
            end
            push!(feat_names, feature.name)
        end

        # Remove the rows that have identical values - we don't want to sample the original instance
        feasible_rows .&= .!identical_rows

        feasible_space[gidx] = combine(
                groupby(
                    data[feasible_rows, feat_names],
                    feat_names),
                nrow => :count
                )

        feasible_space[gidx].distance = distanceFeatureGroup(feasible_space[gidx], orig_instance, group)
    end

    return feasible_space
end
