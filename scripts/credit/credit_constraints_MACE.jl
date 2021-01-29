p = initPLAF()

@PLAF(p, :cf.isMale .== :x.isMale)
@PLAF(p, :cf.isMarried .== :x.isMarried)

@PLAF(p, :cf.EducationLevel .>= :x.EducationLevel)
@PLAF(p, :cf.HasHistoryOfOverduePayments .>= :x.HasHistoryOfOverduePayments)
@PLAF(p, :cf.TotalOverdueCounts .>= :x.TotalOverdueCounts)
@PLAF(p, :cf.TotalMonthsOverdue .>= :x.TotalMonthsOverdue)

# If Education Level increases and Age is not adult, then move to next AgeGroup
@PLAF(p, if cf.EducationLevel .> x.EducationLevel + 1 .&& x.AgeGroup == 1; cf.AgeGroup == x.AgeGroup + 1 end )

# If MonthsWithLowSpendingOverLast6Months increases then MonthsWithHighSpendingOverLast6Months should decrease
@PLAF(p, if cf.MonthsWithLowSpendingOverLast6Months .> x.MonthsWithLowSpendingOverLast6Months
        cf.MonthsWithHighSpendingOverLast6Months .< x.MonthsWithHighSpendingOverLast6Months
    end)

# NOTE: We would like to include the inverse of the last constraint, but this
# would create a cycle in the dependency graph of the constraints. If you would
# like to help us improve this part, please get in touch :)