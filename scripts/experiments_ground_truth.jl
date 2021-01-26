
# Things to consider:
# -- Different number of changed features --> continuous, categorical, different domain sizes
# -- What is the distance? How does it compare to the minimum distance?
# -- # of features changed, and is this equal to the expect number of changes
# -- number of generations required to find this explanation
# -- maybe: experiments with monotonicty  (for later)

# For now, let's do this for the credit dataset

function classifier(instance::DataFrameRow)
    ranges = {:AgeGroup=>123432, }

    if instance.AgeGroup >= 40
        return 1
    else
        norm_distance = average(abs(40 - instance.AgeGroup) / ranges[:AgeGroup], ...)
        return 0.5 - 0.5 * norm_distance
    end
end

instance_to_explain = some_instance_with_ageGroup < 20
explanation, = explain(instance_to_explain, X, p, classifier)

