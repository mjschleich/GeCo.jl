using CSV, Statistics, DataFrames, MLJ, ScikitLearn


const loadData = true
const learnModel = false
const layers = (100,100)

const path = "data/yelp"

if loadData
    include("yelp_data_load.jl")

    serialize(path*"/train_data.bin", X)
    serialize(path*"/train_data_y.bin", y)
else
    X = deserialize(path*"/train_data.bin")
    y = deserialize(path*"/train_data_y.bin")
end

# load the model
@sk_import neural_network: MLPClassifier

if learnModel
    println("Learning MLP model")

    # split the dataset
    train, test = partition(eachindex(y), 0.7, shuffle=true)

    for layer_sizes in [(10,10), (100,), (100,10), (100,100), (200,), (200,10), (200,100), (100,100,10)]

        println("Yelp: Layer Sizes: $(layer_sizes) -- ($(Dates.now()))")
        classifier=MLPClassifier(hidden_layer_sizes=layer_sizes)

        # train
        # ScikitLearn.fit!(classifier,MLJ.matrix(X[train]),vec(collect(Int, y[train])))
        ScikitLearn.fit!(classifier,MLJ.matrix(X),vec(collect(Int, y)))

        ## Evaluation:
        yhat_train = ScikitLearn.predict(classifier, MLJ.matrix(X[train,:]))
        yhat_test = ScikitLearn.predict(classifier, MLJ.matrix(X[test,:]))

        println("Accuracy train data: $(mean(yhat_train .== y[train]))")
        println("Accuracy test data: $(mean(yhat_test .== y[test]))")

        serialize(path*"/mlp_classifier_$(layer_sizes).bin",  classifier)
    end

    # yhat = ScikitLearn.predict(classifier, MLJ.matrix(X))
    # first_neg = findfirst(yhat .!= 1)
    # println(first_neg)
    # orig_instance = X[first_neg,:]
    # partial_classifier = initMLPEval(classifier,orig_instance)
else
    println("Loading MLP model")
    classifier = deserialize(path*"/mlp_classifier_$(layers).bin")
end

include("yelp_constraints.jl");