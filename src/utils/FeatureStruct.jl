using MLJScientificTypes

@enum FeatureConstraint begin
    NOCONSTRAINT
    INCREASING
    DECREASING
end

@enum FeatureType begin
    CONTINUOUS
    CATEGORICAL
    ORDINAL
end

struct Feature
    name::String
    type::FeatureType
    actionable::Bool
    constraint::FeatureConstraint
    range::Float64
end

# struct Feature_new
#     name::String
#     type::FeatureType
#     actionable::Bool
#     constraint::FeatureConstraint
#     range::Float64
# end


struct FeatureGroup_old
    features::Array{Feature, 1}
    names::Array{String, 1}
    indexes::Array{Int64, 1}
    allCategorical::Bool
end

struct FeatureGroup
    features::Tuple{Vararg{Symbol}}
    names::Vector{Symbol}
    indexes::BitVector                  ## TODO: replace indexes accordingly
    allCategorical::Bool
end

function initializeFeatures(file_path::String, data::DataFrame)
    json_file = open(file_path, "r")
    return initializeFeatures(JSON.parse(json_file), data)
end

function initializeFeatures(features::Dict, data::DataFrame)

    if !haskey(features, "features")
        @error("This JSON object does not specify the features of the model.")
    end

    feature_list = Array{Feature,1}()
    onehotFeature = falses(length(features["features"]))
    nameToFeatureMap = Dict()

    for (i,feature) in enumerate(features["features"])

        # Name of the feature
        name = feature["name"]

        # Type of the feature
        feature_type = CONTINUOUS
        ## TODO: Should we use the scitype here ?
        if (feature["type"] == "ordinal")
            feature_type = ORDINAL
        elseif (feature["type"] == "categorical")
            feature_type = CATEGORICAL
        end

        # Constraints of the feature
        actionable = true
        feature_constraint = NOCONSTRAINT

        if haskey(feature, "constraints")
            for constraint in feature["constraints"]
                if constraint == "increasing"
                    feature_constraint = INCREASING
                elseif constraint == "decreasing"
                    feature_constraint = DECREASING
                elseif constraint == "inactionable"
                    actionable = false
                end
            end
        end


        if !(haskey(feature, "onehot") && feature["onehot"])

            # Compute the range of this feature
            ## TODO: Allow the user to define the feature range
            range = maximum(data[:,name]) - minimum(data[:,name])

            # Add to feature list
            push!(feature_list, Feature(name, feature_type, actionable, feature_constraint, range))
            nameToFeatureMap[name] = [feature_list[end]]
        else
            onehotFeature[i] = true

            onehot_group = []
            for colname in String.(names(data))
                if occursin(name*"_", colname)
                    onehot_feature = Feature(colname, CONTINUOUS, actionable, feature_constraint, 1)

                    # Add to feature list
                    push!(feature_list, onehot_feature)
                    push!(onehot_group, onehot_feature)
                end
            end
            nameToFeatureMap[name] = onehot_group
        end
    end

    @assert length(feature_list) == size(data,2) "length(feat_list) ($(length(feature_list))) != size(data,2) ($(size(data,2)))"

    addedFeatures = Set{String}()
    groups = Array{FeatureGroup, 1}()

    ## Generate Feature Groups
    if haskey(features, "groups")
        for group in features["groups"]
            groupFeatures = Feature[]

            allCategorical = true

            for feature in group

                feature in addedFeatures && @error("Each feature can be in at most one group!")

                append!(groupFeatures, nameToFeatureMap[feature])

                if length(nameToFeatureMap[feature]) == 1
                    feat = nameToFeatureMap[feature][1]
                    allCategorical = (feat.type != CONTINUOUS) && (feat.type != ORDINAL)
                end

                push!(addedFeatures, feature)
            end

            union!(addedFeatures, feature.name for feature in groupFeatures)
            push!(groups,
                FeatureGroup(groupFeatures,
                    [feature.name for feature in groupFeatures],
                    [findfirst(isequal(feature), feature_list) for feature in groupFeatures],
                    allCategorical
                    )
                )
        end
    end

    for (idx, feature) in enumerate(feature_list)
        if !(feature.name in addedFeatures)
            push!(groups, FeatureGroup([feature], [feature.name], [idx], (feature.type == CATEGORICAL)))
        end
    end

    return Dict(feature.name => feature for feature in feature_list), groups
end

