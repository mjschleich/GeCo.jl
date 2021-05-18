using CSV, Statistics, DataFrames, MLJ

data = CSV.File("data/german/german_processed.csv") |> DataFrame
# data_y, data = unpack(data, ==("GoodCustomer (label)"), colname -> true);
data_y = data[!, 1]

data = select!(data, Not(1))
for feature in String.(names(data))
    data[!,feature] = convert.(Float64,data[!,feature])
end

# load the model
tree_model = @load RandomForestClassifier pkg=DecisionTree

# change the input to the type they want
Xs = coerce(data,
    :Sex => Count,
    :Age => Count,
    :Credit => MLJ.Continuous,
    :LoanDuration => MLJ.Continuous
)

# change the target to the desired
ys = categorical(data_y)


models(matching(Xs, ys))
tree = machine(tree_model, Xs, ys)

# split the dataset
train, test = partition(eachindex(ys), 0.7, shuffle=true)

# train
fit!(tree, rows=train)

# # get the result
# yhat = MLJ.predict(tree, Xs[test,:]);
orig_instance= Xs[2, :]
X = Xs

p = PLAFProgram()
@PLAF(p, cf.Sex .== x.Sex)
@PLAF(p, cf.Age .>= x.Age)