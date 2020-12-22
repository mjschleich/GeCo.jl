p = initPLAF()

@PLAF(p, :cf.isMale .== :x.isMale)
@PLAF(p, :cf.isMarried .== :x.isMarried)
@PLAF(p, :cf.EducationLevel .>= :x.EducationLevel)
@PLAF(p, :cf.HasHistoryOfOverduePayments .>= :x.HasHistoryOfOverduePayments)

# @PLAF(p, :cf.EducationLevel ∈ (1,2,3))
# @PLAF(p, if :cf.EducationLevel > :x.EducationLevel; :cf.Age > :x.Age + 4 end)

# @GROUP(p, isMale, isMarried)
# @GROUP(p, EducationLevel, Age)

