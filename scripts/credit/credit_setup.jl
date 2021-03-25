using CSV, Statistics, DataFrames, MLJ

path = "data/credit"
data = CSV.File(path*"/credit_processed.csv") |> DataFrame
y, X = unpack(data, ==(:NoDefaultNextMonth), colname -> true);

#for feature in String.(names(data))
#    data[!,feature] = convert.(Float64,data[!,feature])
#end

# load the model
tree_model = @load RandomForestClassifier pkg=DecisionTree
# tree_model = @load DecisionTreeClassifier pkg=DecisionTree
tree_model.max_depth = 3

# change the input to the type they want
coerce!(X,
        :isMale => Count,
        :isMarried => Count,
        :AgeGroup => Count,
        :EducationLevel => Count,
        :MaxBillAmountOverLast6Months => Continuous,
        :MaxPaymentAmountOverLast6Months => Continuous,
        :MonthsWithZeroBalanceOverLast6Months => Continuous,
        :MonthsWithLowSpendingOverLast6Months => Continuous,
        :MonthsWithHighSpendingOverLast6Months => Continuous,
        :MostRecentBillAmount => Continuous,
        :MostRecentPaymentAmount => Continuous,
        :TotalOverdueCounts => Continuous,
        :TotalMonthsOverdue => Continuous,
        :HasHistoryOfOverduePayments => Count,
)

# change the target to the desired
y = categorical(y)

# one-hot encode EducationLevel
# onehot_encoder = OneHotEncoder(; features=[:EducationLevel, :AgeGroup], drop_last=false, ordered_factor=false)
# onehot_machine = machine(onehot_encoder, X)
# fit!(onehot_machine)
# X = MLJ.transform(onehot_machine, X)

# split the dataset
train, test = partition(eachindex(y), 0.7, shuffle=true)
classifier = machine(tree_model, X, y)

# train
MLJ.fit!(classifier, rows=train)

orig_instance = X[14, :]

# using ScikitLearn
# @sk_import tree: DecisionTreeClassifier

# X2 = MLJ.matrix(X)
# classifier = DecisionTreeClassifier()
# ScikitLearn.fit!(classifier, X2, y)
# classifier.tree_.__getstate__()
