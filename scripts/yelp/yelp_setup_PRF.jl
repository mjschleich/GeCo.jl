using CSV, Statistics, DataFrames, MLJ, Serialization

const loadData = false
const learnModel = true
const path = "data/yelp"

if loadData
    include("yelp_data_load.jl")

    serialize(path*"/train_data.bin", X)
    serialize(path*"/train_data_y.bin", y)
else
    X = deserialize(path*"/train_data_less_categ.bin")
    y = deserialize(path*"/train_data_y_less_categ.bin")
end

if learnModel
    println("Learning Partial RF model")

    # load the model
    tree_model = @load RandomForestClassifier pkg=DecisionTree
    tree_model.max_depth = 10

    # split the dataset
    train, test = partition(eachindex(y), 0.7, shuffle=true)
    classifier = machine(tree_model, X, y)

    # train
    MLJ.fit!(classifier, rows=train)

    ## Evaluation:
    yhat_train = MLJ.predict(classifier, X[train,:])
    yhat_test = MLJ.predict(classifier, X[test,:])

    println("Accuracy train data: $(accuracy(mode.(yhat_train), y[train]))")
    println("Accuracy test data: $(accuracy(mode.(yhat_test), y[test]))")

    serialize(path*"/prf_classifier_less_cated.bin",  classifier)

else

    println("Loading Partial RF model")
    classifier = deserialize(path*"/prf_classifier.bin")
end

orig_instance = X[536, :]

# partial_classifier = initPartialRandomForestEval(classifier, orig_instance, 1);
# full_classifier = initRandomForestEval(classifier, orig_instance, 1);

include("yelp_constraints.jl")