using Statistics, DataFrames, MLJ, Serialization

const path = "data/allstate"
const loadData = false

if loadData
    include("allstate_data_load.jl")

    serialize(path*"/train_data.bin", X)
    serialize(path*"/train_data_y.bin", y)
else
    X = deserialize(path*"/train_data.bin")
    y = deserialize(path*"/train_data_y.bin")
end


# load the model
tree_model = @load RandomForestClassifier pkg=DecisionTree
tree_model.n_trees = 500
tree_model.min_samples_leaf = 3
tree_model.max_depth = 10

classifier = machine(tree_model, X, y)

# train
MLJ.fit!(classifier)

## Evaluation:
yhat_train = MLJ.predict(classifier, X[train,:])
yhat_test = MLJ.predict(classifier, X[test,:])

println("Accuracy train data: $(accuracy(mode.(yhat_train), y[train]))")
println("Accuracy test data: $(accuracy(mode.(yhat_test), y[test]))")

# yhat = predict(classifier, X)
# for (i,y) in enumerate(yhat)
#     if mode(y) == false
#         println(i)
#         break;
#     end
# end

orig_instance = X[1,:]
# prf_classifier = initPartialRandomForestEval(mlj_classifier, orig_instance, 1);
# frf_classifier = initRandomForestEval(mlj_classifier, orig_instance, 1);

include("allstate_constraints.jl")