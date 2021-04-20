using CSV, Statistics, DataFrames, MLJ

# load the data set
# for the credit dataset, since the 1 labeled entities is outweighted the 
# 0 labeled entities, we have to do a pre-filter to maintain that about
# half of the entity is labeled for each outcomes
data = CSV.File("data/credit/credit_processed.csv") |> DataFrame
data_y = data[!, 1]

data = select!(data, Not(1))
for feature in String.(names(data))
    data[!,feature] = convert.(Float64,data[!,feature])
end
# load the model
tree_model = @load RandomForestClassifier pkg=DecisionTree

# change the input to the type they want
Xs = coerce(data,
    :isMale => OrderedFactor,
    :isMarried => OrderedFactor,
    :AgeGroup => OrderedFactor,
    :EducationLevel => OrderedFactor,
    :MaxBillAmountOverLast6Months => Continuous,
    :MaxPaymentAmountOverLast6Months => Continuous,
    :MonthsWithZeroBalanceOverLast6Months => OrderedFactor,
    :MonthsWithLowSpendingOverLast6Months => OrderedFactor,
    :MonthsWithHighSpendingOverLast6Months => OrderedFactor,
    :MostRecentBillAmount => Continuous,
    :MostRecentPaymentAmount => Continuous,
    :TotalOverdueCounts => OrderedFactor,
    :TotalMonthsOverdue => OrderedFactor,
    :HasHistoryOfOverduePayments => OrderedFactor,
)

# change the target to the desired
ys = categorical(data_y)


models(matching(Xs, ys))
tree = machine(tree_model, Xs, ys)

# split the dataset
train, test = partition(eachindex(ys), 0.7, shuffle=true)

# train
MLJ.fit!(tree, rows=train)

# get the result
yhat = MLJ.predict(tree, Xs[test,:]);
#save("random_forest_tree_classifier.bson", tree) 
MLJ.save("random_forest_tree_classifier.jlso", tree) 