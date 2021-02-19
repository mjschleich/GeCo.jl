
### ALL CONSTRAINTS & GROUPS

p_all = PLAFProgram()

@GROUP(p_all, s for s in propertynames(X) if contains(string(s), "Relationship"))
@GROUP(p_all, s for s in propertynames(X) if contains(string(s), "Occupation"))
@GROUP(p_all, s for s in propertynames(X) if contains(string(s), "MaritalStatus"))
@GROUP(p_all, s for s in propertynames(X) if contains(string(s), "WorkClass"))
@GROUP(p_all, EducationNumber, EducationLevel)

@PLAF(p_all, cf.Sex .== x.Sex)
@PLAF(p_all, cf.NativeCountry .== x.NativeCountry)

@PLAF(p_all, cf.Age .>= x.Age)
@PLAF(p_all, cf.EducationNumber .>= x.EducationNumber)
@PLAF(p_all, cf.EducationLevel .>= x.EducationLevel)

@PLAF(p_all, cf.MaritalStatus_cat_0 .== x.MaritalStatus_cat_0 &&
    cf.MaritalStatus_cat_1 .== x.MaritalStatus_cat_1  &&
    cf.MaritalStatus_cat_2 .== x.MaritalStatus_cat_2 &&
    cf.MaritalStatus_cat_3 .== x.MaritalStatus_cat_3 &&
    cf.MaritalStatus_cat_4 .== x.MaritalStatus_cat_4 &&
    cf.MaritalStatus_cat_5 .== x.MaritalStatus_cat_5 &&
    cf.MaritalStatus_cat_6 .== x.MaritalStatus_cat_6)

@PLAF(p_all, cf.Relationship_cat_0 .== x.Relationship_cat_0 &&
    cf.Relationship_cat_1 .== x.Relationship_cat_1 &&
    cf.Relationship_cat_2 .== x.Relationship_cat_2 &&
    cf.Relationship_cat_3 .== x.Relationship_cat_3 &&
    cf.Relationship_cat_4 .== x.Relationship_cat_4 &&
    cf.Relationship_cat_5 .== x.Relationship_cat_5)

@PLAF(p_all, if cf.EducationLevel .> x.EducationLevel; cf.Age .>= x.Age + 4 end )


### ALL CONSTRAINT WITHOUT THE IMPLICATIONS

p_no_imp = PLAFProgram(p_all)
empty!(p_no_imp.implications)

### ALL CONSTRAINT WITHOUT THE IMPLICATIONS AND NO INCREASING ONLY CONSTRAINTS

p_no_increasing = PLAFProgram()

@GROUP(p_no_increasing, s for s in propertynames(X) if contains(string(s), "Relationship"))
@GROUP(p_no_increasing, s for s in propertynames(X) if contains(string(s), "Occupation"))
@GROUP(p_no_increasing, s for s in propertynames(X) if contains(string(s), "MaritalStatus"))
@GROUP(p_no_increasing, s for s in propertynames(X) if contains(string(s), "WorkClass"))
@GROUP(p_no_increasing, EducationNumber, EducationLevel)

@PLAF(p_no_increasing, cf.Sex .== x.Sex)
@PLAF(p_no_increasing, cf.NativeCountry .== x.NativeCountry)

@PLAF(p_no_increasing, cf.MaritalStatus_cat_0 .== x.MaritalStatus_cat_0 &&
    cf.MaritalStatus_cat_1 .== x.MaritalStatus_cat_1  &&
    cf.MaritalStatus_cat_2 .== x.MaritalStatus_cat_2 &&
    cf.MaritalStatus_cat_3 .== x.MaritalStatus_cat_3 &&
    cf.MaritalStatus_cat_4 .== x.MaritalStatus_cat_4 &&
    cf.MaritalStatus_cat_5 .== x.MaritalStatus_cat_5 &&
    cf.MaritalStatus_cat_6 .== x.MaritalStatus_cat_6)

@PLAF(p_no_increasing, cf.Relationship_cat_0 .== x.Relationship_cat_0 &&
    cf.Relationship_cat_1 .== x.Relationship_cat_1 &&
    cf.Relationship_cat_2 .== x.Relationship_cat_2 &&
    cf.Relationship_cat_3 .== x.Relationship_cat_3 &&
    cf.Relationship_cat_4 .== x.Relationship_cat_4 &&
    cf.Relationship_cat_5 .== x.Relationship_cat_5)



### ONLY THE GROUPS

p_only_group = PLAFProgram(p_all)
empty!(p_only_group.constraints)
empty!(p_only_group.implications)


### ALL CONSTRAINTS NO GROUPS

p_no_group = PLAFProgram(p_all)
empty!(p_no_group.groups)


### COLLECTIONS

constraint_variatations = [PLAFProgram(), p_all, p_no_imp, p_no_increasing, p_only_group, p_no_group]
constraint_descriptions = ["p_empty", "p_all", "p_no_imp", "p_no_increasing", "p_only_group", "p_no_group"]
