include("fico_model.jl")
using CSV, Statistics, DataFrames, .FicoClassifier

path = "data/fico"
X = CSV.File(path*"/heloc_dataset_v1.csv") |> DataFrame
y = X[!, 1]
X = select!(X, Not(1))
for feature in String.(names(X))
    X[!,feature] = convert.(Float64,X[!,feature])
end
orig_entity = X[6, :]
classifier = FicoClassifier.FICO_CLASSIFIER()