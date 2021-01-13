using CSV, Statistics, DataFrames, MLJ

path = "data/credit"
data = CSV.File(path*"/credit_processed.csv") |> DataFrame
y, X = unpack(data, ==(:NoDefaultNextMonth), colname -> true);

#for feature in String.(names(data))
#    data[!,feature] = convert.(Float64,data[!,feature])
#end

# load the model
tree_model = @load RandomForestClassifier pkg=DecisionTree
tree_model.max_depth = 5

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

# split the dataset
train, test = partition(eachindex(y), 0.7, shuffle=true)
mlj_classifier = machine(tree_model, X, y)

# train
MLJ.fit!(mlj_classifier, rows=train)

orig_instance = X[14, :]

classifier = initPartialRandomForestEval(mlj_classifier, orig_instance, 1);
full_classifier = initRandomForestEval(mlj_classifier, orig_instance, 1);