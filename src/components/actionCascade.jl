

function actionCascade(instance::DataFrameRow, implications::Vector{GroundedImplication}; dataManager::Bool=false)
    validAction = true
    for implication in implications
        if  implication.condition(instance)

            ## Assumption: There is only a single feature group that we need to change
            fspace = implication.sampleSpace

            isempty(fspace) && (validAction = false; break)

            sampled_row = StatsBase.sample(1:nrow(fspace), StatsBase.FrequencyWeights(fspace.count))
            features = implication.conseqFeatures

            instance[features] = fspace[sampled_row, features]
            instance[:mod] .|= implication.conseqFeaturesBitVec
        end
    end

    # !validAction &&  println("This action is not valid")
    return validAction
end


function actionCascadeDataManager()

end