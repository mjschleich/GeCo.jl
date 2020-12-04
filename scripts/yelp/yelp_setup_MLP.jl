using CSV, Statistics, DataFrames, MLJ, ScikitLearn

path = "data/yelp"
data = CSV.File(path*"/yelp_data.csv"; limit=1000000) |> DataFrame
data.high_ranking = Int64.(data.review_stars .> 3)

## Reduce number of cities:
city_gb = combine(groupby( data, :city_id), nrow => :count)
sort!(city_gb, :count, rev=true)
top_cities = city_gb.city_id[1:499]
data.city = [(city in top_cities) ? city : 500 for city in data.city_id]

## Reduce number of categories:
categ_gb = combine(groupby(data, :category_id), nrow => :count)
sort!(categ_gb, :count, rev=true)
top_categories = categ_gb.category_id[1:499]
data.category = [(categ in top_categories) ? categ : 500 for categ in data.category_id]

select!(data, Not([:review_stars,:business_id,:user_id,:review_id,:category_id, :city_id]))

y, X = unpack(data, ==(:high_ranking), colname -> true);
y = categorical(y)

println(y[1:10], vec(collect(Int, y[1:10])))

# change the input to the type they want
coerce!(X,
    :city => Multiclass,
    :state_id => Multiclass,
    :category => Multiclass
)

onehot_columns = [:city, :state_id, :category]

# one-hot encode
onehot_encoder = OneHotEncoder(; features=onehot_columns, drop_last=false, ordered_factor=false)
onehot_machine = machine(onehot_encoder, X)
MLJ.fit!(onehot_machine)
X = MLJ.transform(onehot_machine, X)

println("Learning the model")

# split the dataset
train, test = partition(eachindex(y), 0.7, shuffle=true)

# load the model
@sk_import neural_network: MLPClassifier
mlj_classifier=MLPClassifier()
#mlj_classifier=MLPClassifier(hidden_layer_sizes=(10,10))

# train
ScikitLearn.fit!(mlj_classifier,MLJ.matrix(X[train]),vec(collect(Int, y[train])))

## Evaluation:
yhat_train = ScikitLearn.predict(mlj_classifier, MLJ.matrix(X[train,:]))
yhat_test = ScikitLearn.predict(mlj_classifier, MLJ.matrix(X[test,:]))

println("Accuracy train data: $(mean(yhat_train .== y[train]))")
println("Accuracy test data: $(mean(yhat_test .== y[test]))")

yhat = ScikitLearn.predict(mlj_classifier, MLJ.matrix(X))
first_neg = findfirst(yhat .!= 1)
println(first_neg)

orig_entity = X[first_neg,:]
classifier = initMLPEval(mlj_classifier,orig_entity)


