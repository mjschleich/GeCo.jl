p = initPLAF()

@GROUP(p, workclass_Government, workclass_OtherorUnknown, workclass_Private, workclass_Self_Employed)
@GROUP(p, education_Assoc, education_Bachelors, education_Doctorate, education_HS_grad, education_Masters, education_Prof_school, education_School, education_Some_college)
@GROUP(p, marital_status_Divorced, marital_status_Married, marital_status_Separated, marital_status_Single, marital_status_Widowed)
@GROUP(p, occupation_Blue_Collar, occupation_OtherOrUnknown, occupation_Professional, occupation_Sales, occupation_Service, occupation_White_Collar)
@GROUP(p, race_Other, race_White)
@GROUP(p, gender_Female, gender_Male)

@PLAF(p, :cf.gender_Female .== :x.gender_Female && :cf.gender_Male .== :x.gender_Male)
@PLAF(p, :cf.age .>= :x.age)
@PLAF(p, :cf.race_Other .== :x.race_Other && :cf.race_White .== :x.race_White)

@PLAF(p, :cf.marital_status_Divorced .== :x.marital_status_Divorced,
    :cf.marital_status_Married .== :x.marital_status_Married,
    :cf.marital_status_Separated .== :x.marital_status_Separated,
    :cf.marital_status_Single .== :x.marital_status_Single,
    :cf.marital_status_Widowed .== :x.marital_status_Widowed)


# @PLAF(p, :cf.Relationship_cat_0 .== :x.Relationship_cat_0,
#     :cf.Relationship_cat_1 .== :x.Relationship_cat_1,
#     :cf.Relationship_cat_2 .== :x.Relationship_cat_2,
#     :cf.Relationship_cat_3 .== :x.Relationship_cat_3,
#     :cf.Relationship_cat_4 .== :x.Relationship_cat_4,
#     :cf.Relationship_cat_5 .== :x.Relationship_cat_5)