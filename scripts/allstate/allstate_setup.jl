using CSV, Statistics, DataFrames, MLJ

path = "data/allstate"
data = CSV.File(path*"/train_set.csv"; limit=1000000) |> DataFrame

data.NoClaim = [x == 0.0 ? 1 : 0 for x in data.Claim_Amount]

deletecols!(data, [:Row_ID, :Claim_Amount, :Blind_Submodel, :Household_ID, :NVCat, :OrdCat])

## UNDER SAMPLING TO CREATE A BALANCED DATASET
data = data[shuffle(1:size(data,1)), :]

data_groups = DataFrames.groupby(data, :NoClaim)
data_train = data_groups[1][1:size(data_groups[2],1), :]
append!(data_train, data_groups[2])
data_train = data_train[shuffle(1:nrow(data_train)),:]

# data_falses = data[(data.NoClaim .== false), :]
# data_truess = data[(data.NoClaim .== true), :]\
# X_trues = X[(y .== true), :]
# X_train = X_trues[1:size(X_false,1), :]
# append!(X_train, X_falses)
# X_train = X_train[shuffle(1:nrow(X_train)),:]

y, X = unpack(data_train, ==(:NoClaim), colname -> true);

X.Cat12[ismissing.(X.Cat12)] .= "?"
X.Cat12 = convert.(String,X.Cat12)

onehot_features = [:Cat1,:Cat2,:Cat3,:Cat4,:Cat5,:Cat6,:Cat7,:Cat8,:Cat9,:Cat10,:Cat11,:Cat12,:Blind_Make, :Blind_Model]

coerce!(X,
    :Cat1 => Multiclass,
    :Cat2 => Multiclass,
    :Cat3 => Multiclass,
    :Cat4 => Multiclass,
    :Cat5 => Multiclass,
    :Cat6 => Multiclass,
    :Cat7 => Multiclass,
    :Cat8 => Multiclass,
    :Cat9 => Multiclass,
    :Cat10 => Multiclass,
    :Cat11 => Multiclass,
    :Cat12 => Multiclass,
    :Blind_Make => Multiclass,
    :Blind_Model => Multiclass
    )

# change the target to the desired
y = categorical(y)

# split the dataset
train, test = partition(eachindex(y), 0.7, shuffle=true)

# one-hot encode EducationLevel
onehot_encoder = OneHotEncoder(; features=onehot_features, drop_last=false, ordered_factor=false)
onehot_machine = machine(onehot_encoder, X)
fit!(onehot_machine)
X = MLJ.transform(onehot_machine, X)

# load the model
tree_model = @load RandomForestClassifier pkg=DecisionTree
tree_model.n_trees = 500
tree_model.min_samples_leaf = 3
tree_model.max_depth = 50

classifier = machine(tree_model, X, y)

# train
# MLJ.fit!(classifier, rows=train)
MLJ.fit!(classifier)

## Evaluation:
yhat_train = predict(classifier, X[train,:])
yhat_test = predict(classifier, X[test,:])

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