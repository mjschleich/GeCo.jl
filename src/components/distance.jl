# Implementation of the distance measure for the genetic algorithm

@inline absolute(val, orig_val, range)::Float64 = abs(val - orig_val) / range

## Computes the distance between two entities
function distance(entity_1::DataFrameRow, entity_2::DataFrameRow, features::Array{Feature, 1},feature_distance_abs; norm_ratio::Array{Float64,1}=default_norm_ratio)::Float64

    # feature_distance_abs = Array{Float64,1}(undef, length(features))

    # compute the normalized absolute distance for each feature
    for (index, feature) in enumerate(features)
        # check for the type of the feature
        if feature.type == CONTINUOUS || feature.type == ORDINAL
            # if numerical, get the absolute difference and then divided by the range of the feature
            feature_distance_abs[index] = absolute(get(entity_1, feature.name, entity_2[feature.name]), entity_2[feature.name], feature.range)
        elseif feature.type == CATEGORICAL
            # for the categorical -- 1 for they are not same and 0 for same
            feature_distance_abs[index] = (get(entity_1, feature.name, entity_2[feature.name]) == entity_2[feature.name] ? 0 : 1)
        else
            @error "We should never get here! " * feature.type
            println(typeof(feature.type), " -- ", typeof(FeatureStruct.CONTINUOUS))
            break
        end
    end

    feature_len = length(features)

    # sum up to get the true distance for the two entity
    zero_norm_distance = count(i->(i!=0), feature_distance_abs) / feature_len
    one_norm_distance = sum(feature_distance_abs) / feature_len
    two_norm_distance = sqrt(sum(distance * distance for distance in feature_distance_abs) / feature_len )
    infty_norm_distance = maximum(feature_distance_abs) / feature_len

    return zero_norm_distance * norm_ratio[1] + one_norm_distance * norm_ratio[2] +
        two_norm_distance * norm_ratio[3] + infty_norm_distance * norm_ratio[4]
end

function distance(entity_1::AbstractVector{Float64}, entity_2::AbstractVector{Float64}, features::Array{Feature, 1}, feature_distance_abs::Array{Float64,1};
    norm_ratio::Array{Float64,1}=default_norm_ratio)::Float64

    # compute the normalized absolute distance for each feature
    for (index, feature) in enumerate(features)
        # check for the type of the feature
        if feature.type == CONTINUOUS || feature.type == ORDINAL
            # if numerical, get the absolute difference and then divided by the range of the feature
            feature_distance_abs[index] = abs(entity_1[index] - entity_2[index]) / feature.range
        elseif feature.type == CATEGORICAL
            # for the categorical -- 1 for they are not same and 0 for same
            feature_distance_abs[index] = (entity_1[index] == entity_2[index] ? 0 : 1)
        else
            @error "We should never get here! " * feature.type
            println(typeof(feature.type), " -- ", typeof(FeatureStruct.CONTINUOUS))
            break
        end
    end

    feature_len = length(features)

    # sum up to get the true distance for the two entity
    zero_norm_distance = count(i->(i!=0), feature_distance_abs) / feature_len
    one_norm_distance = sum(feature_distance_abs) / feature_len
    two_norm_distance = sqrt(sum(distance * distance for distance in feature_distance_abs) / feature_len )
    infty_norm_distance = maximum(feature_distance_abs) / feature_len

    return zero_norm_distance * norm_ratio[1] + one_norm_distance * norm_ratio[2] +
        two_norm_distance * norm_ratio[3] + infty_norm_distance * norm_ratio[4]
end

function distance(df::DataFrame, orig_entity::DataFrameRow, feature_dict::Dict{String, Feature};
    norm_ratio::Array{Float64,1}=default_norm_ratio)::Array{Float64,1}
    distance_temp = zeros(Float64, 4*size(df,1))
    distance(df,orig_entity,feature_dict,distance_temp,norm_ratio=norm_ratio)
end

function distance(df::DataFrame, orig_entity::DataFrameRow, feature_dict::Dict{String, Feature}, distance_temp::Array{Float64,1};
    norm_ratio::Array{Float64,1}=default_norm_ratio)::Array{Float64,1}

    # distance_temp = zeros(Float64, 4, size(df,1))
    for i = 1:(4*size(df,1))
        distance_temp[i] = 0.0
    end

    # compute the normalized absolute distance for each feature
    for pname in propertynames(df)
        # ad hoc hack
        pname in (:generation, :score, :outc, :mod, :estcf) && continue
        feature = feature_dict[String(pname)]
        # for partial entity
        try
            if feature.type == CONTINUOUS ||  feature.type == ORDINAL
                range = feature.range

                orig_val::Float64 = orig_entity[pname]
                col::Array{Float64,1} = df[!, pname]

                row_index = 0
                for val in col

                    # if numerical, get the absolute difference and then divided by the range of the feature
                    diff = abs(val - orig_val) / range ## absolute(val, orig_val, range)

                    distance_temp[row_index + 1] += (diff != 0.0)                 ## zero norm
                    distance_temp[row_index + 2] += diff                          ## one norm
                    distance_temp[row_index + 3] += diff * diff                   ## two norm
                    distance_temp[row_index + 4] = max(distance_temp[row_index + 4], diff)   ## inf norm

                    row_index += 4
                end
            elseif feature.type == CATEGORICAL

                orig_categ_val::Int64 = orig_entity[feature.name]
                categ_col::Array{Int64,1} = df[!, feature.name]

                row_index = 0
                for val in categ_col

                    # for the categorical -- 1 for they are not same and 0 for same
                    diff::Bool = (val != orig_categ_val)

                    distance_temp[row_index + 1] += diff                          ## zero norm
                    distance_temp[row_index + 2] += diff                          ## one norm
                    distance_temp[row_index + 3] += diff                          ## two norm
                    distance_temp[row_index + 4] = max(distance_temp[row_index + 4], diff)   ## inf norm

                    row_index += 4
                end
            else
                error("Feature type is not defined: " * feature.type * " -- " * typeof(feature.type))
                return
            end
        catch
        end
    end

    feature_len = length(feature_dict)

    return Float64[
        norm_ratio[1] * (distance_temp[(row * 4) + 1] / feature_len) +
        norm_ratio[2] * (distance_temp[(row * 4) + 2] / feature_len) +
        norm_ratio[3] * (sqrt(distance_temp[(row * 4) + 3] / feature_len))  +
        norm_ratio[4] * (distance_temp[(row * 4) + 4] / feature_len)
        for row in 0:(nrow(df)-1)
    ]
end


function distanceFeatureGroup(data::DataFrame, orig_entity::DataFrameRow, group::FeatureGroup)::Vector{Float64}

    output = zeros(size(data,1))

    for feature in group.features
        fname = feature.name
        fsym = Symbol(feature.name)

        if feature.type == CONTINUOUS || feature.type == ORDINAL
            orig_val::Float64 = orig_entity[fsym]

            col::Vector{Float64} = data[!, fsym]
            for (idx, val) in enumerate(col)
                output[idx] += abs(orig_val - val) / feature.range
            end
        elseif feature.type == CATEGORICAL
            categ_val::Int64 = orig_entity[fname]::Int64

            categ_col::Vector{Int64} = data[!, fsym]
            for (idx, val) in enumerate(categ_col)
                output[idx] += (categ_val != val)
            end
        end
    end

    return output
end

function distanceFeatureGroup(data::DataFrameRow, orig_entity::DataFrameRow, group::FeatureGroup)::Float64

    output = 0.0
    for feature in group.features
        fname = feature.name
        if feature.type == CONTINUOUS || feature.type == ORDINAL
            output += abs(orig_entity[fname] - data[fname]) / feature.range     ## TODO: Check for type consistentcy here
        elseif feature.type == CATEGORICAL
            output += (orig_entity[fname] != data[fname])
        end
    end
    return output
end


# check the clostest feature for the first entity
function findClosest(data, orig_entity, features;
    label_name=:pred, desired_class::Int64=1, norm_ratio::Array{Float64,1}=[0.0,1.0,0.0,0.0])

    selected_data = data[data[!, label_name] .== desired_class, :]
    distances = distance(selected_data, orig_entity, features; norm_ratio=norm_ratio)
    row = argmin(distances)
    return selected_data[row, :], distances[row]
end

function findMinimumObservable(data, orig_instance, features;
    check_feasibility::Bool=false,
    desired_class::Int64=1,
    norm_ratio::Array{Float64,1}=[0.0,1.0,0.0,0.0],
    label_name=:preds,
    distance_temp=Array{Float64,1}())

    selected_data = data[data[!, label_name] .== desired_class, :]

    if check_feasibility
        feasible_rows = trues(nrow(selected_data))

        # TODO: we do not consider feature groups here, should we?
        for (fname,feature) in features
            if !feature.actionable
                feasible_rows .&= (selected_data[:, feature.name] .== orig_instance[feature.name])
            elseif feature.constraint == INCREASING
                feasible_rows .&= (selected_data[:, feature.name] .>= orig_instance[feature.name])
            elseif feature.constraint == DECREASING
                feasible_rows .&= (selected_data[:, feature.name] .<= orig_instance[feature.name])
            end
        end
        selected_data = selected_data[feasible_rows, :]
    end

    if isempty(selected_data)
        return nothing, nothing
    end

    if length(distance_temp) < nrow(selected_data) * 4
        resize!(distance_temp, nrow(selected_data) * 4)
    end

    distances = distance(selected_data[!, Not(:pred)], orig_instance, features, distance_temp; norm_ratio=norm_ratio)
    row = argmin(distances)
    return selected_data[row, :], distances[row]
end

function findObservable(data, orig_instance, features;
    check_feasibility::Bool=false,
    desired_class::Int64=1,
    label_name=:preds)

    selected_data = data[data[!, label_name] .== desired_class, :]

    if check_feasibility
        feasible_rows = trues(nrow(selected_data))
        #print(size(filter(x -> x == true, feasible_rows)))
        # TODO: we do not consider feature groups here, should we?
        for (fname,feature) in features

            if !feature.actionable
                feasible_rows .&= (selected_data[:, feature.name] .== orig_instance[feature.name])
            elseif feature.constraint == INCREASING
                feasible_rows .&= (selected_data[:, feature.name] .>= orig_instance[feature.name])
            elseif feature.constraint == DECREASING
                feasible_rows .&= (selected_data[:, feature.name] .<= orig_instance[feature.name])
            end
            # print(feature)
            # println(size(filter(x -> x == true, feasible_rows)))
        end

        selected_data = selected_data[feasible_rows, :]
    end

    if isempty(selected_data)
        return nothing
    end
    return selected_data
end