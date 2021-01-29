using Statistics, DataFrames, MLJ, ScikitLearn, Serialization

const loadData = true
const path = "data/allstate"

if loadData
    include("allstate_data_load.jl")

    serialize(path*"/train_data.bin", X)
    serialize(path*"/train_data_y.bin", y)
else
    X = deserialize(path*"/train_data.bin")
    y = deserialize(path*"/train_data_y.bin")
end

# load the model
@sk_import neural_network: MLPClassifier

# split the dataset
train, test = partition(eachindex(y), 0.7, shuffle=true)

for layer_sizes in [(10,10), (100,), (100,10), (100,100), (200,), (200,10), (200,100)]

    println("Allstate: Layer Sizes: $(layer_sizes) -- ($(Dates.now()))")

    #mlj_classifier=MLPClassifier(hidden_layer_sizes=(200,100,100,))
    classifier=MLPClassifier(hidden_layer_sizes=layer_sizes)

    # Training
    ScikitLearn.fit!(classifier, MLJ.matrix(X), vec(collect(Int, y)))

    ## Evaluation:
    yhat_train = ScikitLearn.predict(classifier, MLJ.matrix(X[train,:]))
    yhat_test = ScikitLearn.predict(classifier, MLJ.matrix(X[test,:]))

    println("Accuracy train data: $(mean(yhat_train .== y[train])) -- ($(Dates.now()))")
    println("Accuracy test data: $(mean(yhat_test .== y[test]))\n")

    serialize(path*"/mlp_classifier_$(layer_sizes).bin",  classifer)
end

# yhat = ScikitLearn.predict(classifier, MLJ.matrix(X))
# first_neg = findfirst(yhat .!= 1)
# println(first_neg)
# orig_instance = X[first_neg,:]
# partial_classifier = initMLPEval(mlj_classifier,orig_instance)

include("allstate_constraints.jl")