p_all = initPLAF()

@PLAF(p_all, :cf.isMale .== :x.isMale)
@PLAF(p_all, :cf.isMarried .== :x.isMarried)

@PLAF(p_all, :cf.AgeGroup .>= :x.AgeGroup)
@PLAF(p_all, :cf.EducationLevel .>= :x.EducationLevel)
@PLAF(p_all, :cf.HasHistoryOfOverduePayments .>= :x.HasHistoryOfOverduePayments)

@PLAF(p_all, :cf.TotalOverdueCounts .>= :x.TotalOverdueCounts)
@PLAF(p_all, :cf.TotalMonthsOverdue .>= :x.TotalMonthsOverdue)

# If Education Level increases and Age is not adult, then move to next AgeGroup
@PLAF(p_all, if cf.EducationLevel .> x.EducationLevel + 1 && x.AgeGroup .< 2; cf.AgeGroup == 2 end )

# If MonthsWithLowSpendingOverLast6Months increases then MonthsWithHighSpendingOverLast6Months should decrease
@PLAF(p_all, if cf.MonthsWithLowSpendingOverLast6Months .> x.MonthsWithLowSpendingOverLast6Months
        cf.MonthsWithHighSpendingOverLast6Months .< x.MonthsWithHighSpendingOverLast6Months
    end)



### NO IMPLICATIONS

p_no_imp = initPLAF(p_all)
empty!(p_no_imp.implications)



### NO INCREASING

p_no_increasing = initPLAF()

@PLAF(p_no_increasing, :cf.isMale .== :x.isMale)
@PLAF(p_no_increasing, :cf.isMarried .== :x.isMarried)


### COLLECTIONS

constraint_variations = [PLAFProgram(), p_all, p_no_imp, p_no_increasing]
constraint_descriptions = ["p_empty", "p_all", "p_no_imp", "p_no_increasing"]


