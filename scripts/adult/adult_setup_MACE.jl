using CSV, Statistics, DataFrames, MLJ,  PyCall, ScikitLearn

path = "data/adult"
data = CSV.File(path*"/adult_data_mace.csv") |> DataFrame

data = select!(data, Not(:Column1))

y, X = unpack(data, ==(:Label), colname -> true);

# change the input to the type they want
coerce!(X,
    :CapitalGain => Continuous,
    :Sex            => Count,
    :HoursPerWeek => Continuous,
    :CapitalLoss   => Continuous,
    :EducationNumber => Count,
    :NativeCountry => Count,
    :EducationLevel => Count,
    :Age            => Count,
    # :marital_status => Multiclass,
    # :Occupation     => Multiclass,
    # :Relationship   => Multiclass,
    # :workclass      => Multiclass
)

# change the target to the desired
y = categorical(y)

# onehot_columns = [:workclass, :marital_status, :occupation, :relationship, :race]
# one-hot encode EducationLevel
# onehot_encoder = OneHotEncoder(; features=onehot_columns, drop_last=false, ordered_factor=false)
# onehot_machine = machine(onehot_encoder, X)
# MLJ.fit!(onehot_machine)
# X = MLJ.transform(onehot_machine, X)

# # split the dataset
# train, test = partition(eachindex(y), 0.7, shuffle=true)
# classifier = machine(tree_model, X, y)

# train
# MLJ.fit!(classifier, rows=train)

pickle = pyimport("pickle")
classifier = pickle.load(pybuiltin("open")("scripts/mace_models/adult_model.pickle","rb"))

orig_entity = X[6, :]