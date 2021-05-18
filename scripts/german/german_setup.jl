using BayesNets, Random, CSV, Statistics, DataFrames
data = CSV.File("data/german/german_processed.csv") |> DataFrame

# all conditional probably distrubutions
cpdSex = fit(StaticCPD{Categorical}, data, :Sex)
cpdAge = fit(StaticCPD{DiscreteNonParametric}, data, :Age)
cpdCredit = fit(LinearGaussianCPD, data, :Credit, [:Age, :Sex])
cpdDuration = fit(LinearGaussianCPD, data, :LoanDuration, [:Credit])

# the Bays Network
bnCredit = BayesNet([cpdAge, cpdSex, cpdCredit, cpdDuration])

include("./german_classifier.jl");