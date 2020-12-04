using CSV, Statistics, DataFrames, MLJ, ScikitLearn

path = "data/credit"
data = CSV.File(path*"/credit_processed.csv") |> DataFrame
y, X = unpack(data, ==(:NoDefaultNextMonth), colname -> true);

#for feature in String.(names(data))
#    data[!,feature] = convert.(Float64,data[!,feature])
#end

# load the model
@sk_import neural_network: MLPClassifier
# it's not mlj tho
mlj_classifier=MLPClassifier()


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

# # train
ScikitLearn.fit!(mlj_classifier, MLJ.matrix(X[train,:]), vec(collect(Int, y[train,:])))

orig_entity = X[14, :]

classifier = initMLPEval(mlj_classifier,orig_entity)
