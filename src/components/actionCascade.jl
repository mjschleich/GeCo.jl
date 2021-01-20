

function actionCascade(instance::DataFrameRow, implications::Vector{GroundedImplication}; dataManager::Bool=false)

    validAction = true
    for implication in implications

        ## TODO: Do we need to include: any(instance[:mod] .& implication.condFeatures) in the if statement?
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

    return validAction
end


function actionCascadeDataManager()

end