using CSV, Statistics, DataFrames, MLJ

path = "data/adult"
data = CSV.File(path*"/adult_processed.csv") |> DataFrame

y, X = unpack(data, ==(:income), colname -> true);

# load the model
tree_model = @load RandomForestClassifier pkg=DecisionTree
tree_model.max_depth = 3

# change the input to the type they want
coerce!(X,
    :race => Multiclass,
    :fnlwgt => Continuous,
    :capital_gain => Continuous,
    :occupation     => Multiclass,
    :relationship   => Multiclass,
    :sex            => Multiclass,
    :hours_per_week => Continuous,
    :capital_loss   => Continuous,
    :education_num  => Continuous,
    :native_country => Multiclass,
    :education      => Multiclass,
    :marital_status => Multiclass,
    :age            => Continuous,
    :workclass      => Multiclass
)

# change the target to the desired
y = categorical(y)

onehot_columns = [:workclass, :education, :marital_status, :occupation, :relationship, :sex, :race, :native_country]

# one-hot encode EducationLevel
onehot_encoder = OneHotEncoder(; features=onehot_columns, drop_last=false, ordered_factor=false)
onehot_machine = machine(onehot_encoder, X)
MLJ.fit!(onehot_machine)
X = MLJ.transform(onehot_machine, X)

# split the dataset
train, test = partition(eachindex(y), 0.7, shuffle=true)
classifier = machine(tree_model, X, y)

# train
MLJ.fit!(classifier, rows=train)

orig_instance = X[6, :]
