using CSV, Statistics, DataFrames, MLJ,  PyCall
torch = pyimport("torch")

path = "data/adult"
X = CSV.File(path*"/adult_data_dice.csv") |> DataFrame

coerce!(X,
        :age => Continuous,
        :hours_per_week => Continuous,
        :workclass_Government => Count,
        :workclass_OtherorUnknown => Count,
        :workclass_Private => Count,
        :workclass_Self_Employed => Count,
        :education_Assoc => Count,
        :education_Bachelors => Count,
        :education_Doctorate => Count,
        :education_HS_grad => Count,
        :education_Masters => Count,
        :education_Prof_school => Count,
        :education_School => Count,
        :education_Some_college => Count,
        :marital_status_Divorced => Count,
        :marital_status_Married => Count,
        :marital_status_Separated => Count,
        :marital_status_Single => Count,
        :marital_status_Widowed => Count,
        :occupation_Blue_Collar => Count,
        :occupation_OtherOrUnknown => Count,
        :occupation_Professional => Count,
        :occupation_Sales => Count,
        :occupation_Service => Count,
        :occupation_White_Collar => Count,
        :race_Other => Count,
        :race_White => Count,
        :gender_Female => Count,
        :gender_Male => Count
)

classifier = torch.load("./data/adult/adult_dice_model.pth")
orig_instance = X[6,:]

include("adult_constraints_DICE.jl")

#in = torch.tensor(convert(Array, orig_instance)).float()
#label = classifier(in).float()
#label.detach().numpy()[1]
