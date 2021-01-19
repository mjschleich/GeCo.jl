

function actionCascade(instance::DataFrameRow, implications::Vector{GroundedImplication}; dataManager::Bool=false)

    validAction = true
    for implication in implications

        ## TODO: we need to check the body of the if conditon is not already satisfied
        if any(instance[:mod] .& implication.condFeatures) && implication.condition(instance)
            println("HERE ... ")

            ## Assumption: There may be only a single feature group that we need to change
            fspace = implication.sampleSpace

            isempty(fspace) && (validAction = false; break)

            sampled_row = StatsBase.sample(1:nrow(fspace), StatsBase.FrequencyWeights(fspace.count))
            features = implication.conseqFeatures

            println("Replacing value of instance with: ", fspace[sampled_row, features])
            instance[features] = fspace[sampled_row, features]

            !validAction && break;
            # instance[:mod] .|= constraint.changedFeatures

        else
            println("All constraints satisfied ...")
        end
    end

    return validAction
end


function actionCascadeDataManager()

end