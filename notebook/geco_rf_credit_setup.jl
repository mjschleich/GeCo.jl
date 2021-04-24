# import Pkg; Pkg.activate(".")
using GeCo, CSV, Statistics, DataFrames, MLJ, PyCall, ScikitLearn

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

@load RandomForestClassifier pkg=DecisionTree
classifier = machine("random_forest_tree_classifier.jlso")
orig_instance = X[110, :]

## PLAF Constraints:
p = PLAFProgram()

@PLAF(p, cf.isMale .== x.isMale)
@PLAF(p, cf.isMarried .== x.isMarried)

@PLAF(p, cf.AgeGroup .>= x.AgeGroup)
@PLAF(p, cf.EducationLevel .>= x.EducationLevel)
@PLAF(p, cf.HasHistoryOfOverduePayments .>= x.HasHistoryOfOverduePayments)
@PLAF(p, cf.TotalOverdueCounts .>= x.TotalOverdueCounts)
@PLAF(p, cf.TotalMonthsOverdue .>= x.TotalMonthsOverdue)

# If Education Level increases and Age is not adult, then move to next AgeGroup
@PLAF(p, if cf.EducationLevel .> x.EducationLevel + 1 && x.AgeGroup .< 2; cf.AgeGroup == 2 end )

# If MonthsWithLowSpendingOverLast6Months increases then MonthsWithHighSpendingOverLast6Months should decrease
@PLAF(p, if cf.MonthsWithLowSpendingOverLast6Months .> x.MonthsWithLowSpendingOverLast6Months
        cf.MonthsWithHighSpendingOverLast6Months .< x.MonthsWithHighSpendingOverLast6Months
    end)

## Run once to avoid compilation overhead
explanation,  = @time explain(orig_instance, X, p, classifier);
