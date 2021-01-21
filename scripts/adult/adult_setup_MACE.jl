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
)

y = categorical(y)

pickle = pyimport("pickle")
classifier = pickle.load(pybuiltin("open")("scripts/mace_models/adult_model.pickle","rb"))

include("adult_constraints_MACE.jl");

orig_instance = X[6, :]
