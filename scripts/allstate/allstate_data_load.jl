using CSV, Statistics, DataFrames, MLJ

#data = CSV.File(path*"/train_set.csv"; limit=1000000) |> DataFrame
data = CSV.File(path*"/train_set.csv") |> DataFrame

## Reduce number of BlindModels:
model_gb = combine(groupby( data, :Blind_Model), nrow => :count)
sort!(model_gb, :count, rev=true)
top_models = model_gb.Blind_Model[1:399]
data.Model = [(model in top_models) ? model : "Other" for model in data.Blind_Model]

data.NoClaim = [x == 0.0 ? 1 : 0 for x in data.Claim_Amount]

deletecols!(data, [:Row_ID, :Claim_Amount, :Blind_Submodel, :Household_ID, :NVCat, :OrdCat, :Blind_Model])

## UNDER SAMPLING TO CREATE A BALANCED DATASET
data = data[shuffle(1:size(data,1)), :]

data_groups = groupby(data, :NoClaim)
data_train = data_groups[1][1:size(data_groups[2],1), :]
append!(data_train, data_groups[2])
data_train = data_train[shuffle(1:nrow(data_train)),:]

y, X = unpack(data_train, ==(:NoClaim), colname -> true);

X.Cat12[ismissing.(X.Cat12)] .= "?"
X.Cat12 = convert.(String,X.Cat12)

onehot_features = [:Cat1,:Cat2,:Cat3,:Cat4,:Cat5,:Cat6,:Cat7,:Cat8,:Cat9,:Cat10,:Cat11,:Cat12,:Blind_Make,:Model]

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
    :Model => Multiclass
    )

# change the target to the desired
y = categorical(y)

# one-hot encode EducationLevel
onehot_encoder = OneHotEncoder(; features=onehot_features, drop_last=false, ordered_factor=false)
onehot_machine = machine(onehot_encoder, X)
MLJ.fit!(onehot_machine)
X = MLJ.transform(onehot_machine, X)
