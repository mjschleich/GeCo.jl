p = initPLAF()

@PLAF(p, :cf.isMale .== :x.isMale)
@PLAF(p, :cf.isMarried .== :x.isMarried)
@PLAF(p, :cf.EducationLevel .>= :x.EducationLevel)
@PLAF(p, :cf.HasHistoryOfOverduePayments .>= :x.HasHistoryOfOverduePayments)

