using CSV, Statistics, DataFrames, MLJ, PyCall, ScikitLearn

path = "data/credit"
data = CSV.File(path*"/credit_processed.csv") |> DataFrame
y, X = unpack(data, ==(:NoDefaultNextMonth), colname -> true);

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
# MLJ.fit!(onehot_machine)
# X = MLJ.transform(onehot_machine, X)

pickle = pyimport("pickle")
classifier = pickle.load(pybuiltin("open")("scripts/mace_models/credit_model.pickle","rb"))
# classifier.tree_.__getstate__()

orig_entity = X[110, :]

include("credit_constraints_MACE.jl")
