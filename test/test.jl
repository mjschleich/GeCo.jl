
using Pkg; Pkg.activate(".")
using GeCo, Test
using CSV, Statistics, DataFrames, MLJ, JSON 
# include("../src/utils/FeatureStruct.jl")
# include("../src/components/feasibleSpace.jl")
# include("../src/components/distance.jl")
# using Pkg; Pkg.activate(".")
# using GeCo, Test

@testset "Distance Test" begin
    # the set up for the distance test for credit dataset
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
    features, groups = initializeFeatures(path*"/data_info.json", X)
    orig_entity = X[1,:]
    feasibleSpace(orig_entity, groups, X)
    feature_distance_abs = Array{Float64,1}(undef, length(features))

    # create the features for the array based 
    features_array = Array{Feature,1}(undef, length(features))
    index = 1
    for value in values(features)
        features_array[index] = value
        index += 1
    end

    test_entity = DataFrame(orig_entity)
    l0_norm = [1.0, 0.0, 0.0, 0.0]
    l1_norm = [0.0, 1.0, 0.0, 0.0]
    combined_norm = [0.25, 0.25, 0.25, 0.25]

    # simple test that the distance of the same entity shuold be 0
    @test distance(test_entity, orig_entity, features; norm_ratio = combined_norm) == [0.0]
    @test distance(test_entity[1,:], orig_entity, features_array, feature_distance_abs; norm_ratio = combined_norm) == 0.0

    ## simple test of change one categorical feature for both distance functions 
    test_entity[1, "isMale"] = 1
    #for l0 norm
    @test distance(test_entity, orig_entity, features; norm_ratio = l0_norm) == [1.0/length(features)]
    @test distance(test_entity[1,:], orig_entity, features_array, feature_distance_abs; norm_ratio = l0_norm) == 1.0/length(features)
    # for l1 norm
    @test distance(test_entity, orig_entity, features; norm_ratio = l1_norm) == [1.0/length(features)]
    @test distance(test_entity[1,:], orig_entity, features_array, feature_distance_abs; norm_ratio = l1_norm) == 1.0/length(features)
    #for combined norm
    @test distance(test_entity, orig_entity, features; norm_ratio = combined_norm) == [1.0/length(features)*3/4 + sqrt(1.0/length(features))/4 ]
    @test distance(test_entity[1,:], orig_entity, features_array, feature_distance_abs; norm_ratio = combined_norm) == 1.0/length(features)*3/4 + sqrt(1.0/length(features))/4
    test_entity[1, "isMale"] = 0

    ## simple test of change one Continuous feature for both distance functions 
    test_entity[1, "MaxBillAmountOverLast6Months"] += 10
    #for l0 norm
    @test distance(test_entity, orig_entity, features; norm_ratio = l0_norm) == [1/length(features)]
    @test distance(test_entity[1,:], orig_entity, features_array, feature_distance_abs; norm_ratio = l0_norm) == 1.0/length(features)
    # for l1 norm
    @test distance(test_entity, orig_entity, features; norm_ratio = l1_norm) == [10.0/50810/length(features)]
    @test distance(test_entity[1,:], orig_entity, features_array, feature_distance_abs; norm_ratio = l1_norm) == 10.0/50810/length(features)
    #for combined norm
    @test distance(test_entity, orig_entity, features; norm_ratio = combined_norm) == [1.0/length(features)/4 + 10.0/50810/length(features)/2 + sqrt(10.0/50810*10.0/50810/length(features))/4 ]
    @test distance(test_entity[1,:], orig_entity, features_array, feature_distance_abs; norm_ratio = combined_norm) == 1.0/length(features)/4 + 10.0/50810/length(features)/2 + sqrt(10.0/50810*10.0/50810/length(features))/4
    test_entity[1, "MaxBillAmountOverLast6Months"] -= 10

    ## simple test of changing several categorical (2) and Continuous (2) features
    test_entity[1, "isMale"] = 1
    test_entity[1, "isMarried"] = 0
    test_entity[1, "MaxBillAmountOverLast6Months"] += 10
    test_entity[1, "EducationLevel"] += 2
    #for l0 norm
    @test distance(test_entity, orig_entity, features; norm_ratio = l0_norm) == [4/length(features)]
    @test distance(test_entity[1,:], orig_entity, features_array, feature_distance_abs; norm_ratio = l0_norm) == 4.0/length(features)
    # for l1 norm
    @test distance(test_entity, orig_entity, features; norm_ratio = l1_norm) == [(2+2/3+10.0/50810)/length(features)]
    @test distance(test_entity[1,:], orig_entity, features_array, feature_distance_abs; norm_ratio = l1_norm) == (2+2/3+10.0/50810)/length(features)
    #for combined norm
    @test distance(test_entity, orig_entity, features; norm_ratio = combined_norm) == [4.0/length(features)/4 + (2+2/3+10.0/50810)/length(features)/4 + sqrt((2+2*2/3/3+10.0/50810*10.0/50810)/length(features))/4 + 1.0/length(features)/4]
    @test distance(test_entity[1,:], orig_entity, features_array, feature_distance_abs; norm_ratio = combined_norm) == 4.0/length(features)/4 + (2+2/3+10.0/50810)/length(features)/4 + sqrt((2+2*2/3/3+10.0/50810*10.0/50810)/length(features))/4 + 1.0/length(features)/4
    test_entity[1, "MaxBillAmountOverLast6Months"] -= 10


    ### tests mainly check whether the two measures return the same result

    ## check for all entities compare with the first entity (labeled as good)
    #for l1 norm
    single_distance_result = Array{Float64,1}(undef, size(X,1))
    combined_distance_result = distance(X, orig_entity, features; norm_ratio = l1_norm)
    for index in 1:size(X,1)
        single_distance_result[index] = distance(X[index,:], orig_entity, features_array, feature_distance_abs; norm_ratio = l1_norm)
        single_distance_result[index] = floor(single_distance_result[index]*10000000000)/10000000000.0
        combined_distance_result[index] = floor(combined_distance_result[index]*10000000000)/10000000000.0
        # if single_distance_result[index] != combined_distance_result[index]
        #     println("$index: $(single_distance_result[index]) with $(combined_distance_result[index])" )
        # end
    end
    @test combined_distance_result== single_distance_result
    # for combined norm
    combined_distance_result = distance(X, orig_entity, features; norm_ratio = combined_norm)
    for index in 1:size(X,1)
        single_distance_result[index] = distance(X[index,:], orig_entity, features_array, feature_distance_abs; norm_ratio = combined_norm)
        single_distance_result[index] = floor(single_distance_result[index]*10000000000)/10000000000.0
        combined_distance_result[index] = floor(combined_distance_result[index]*10000000000)/10000000000.0
        # if single_distance_result[index] != combined_distance_result[index]
        #     println("$index: $(single_distance_result[index]) with $(combined_distance_result[index])" )
        # end
    end
    @test combined_distance_result== single_distance_result

    ## check for all entities compare with the 14 entity (labeled as bad)
    orig_entity = X[14,:]
    #for l1 norm
    combined_distance_result = distance(X, orig_entity, features; norm_ratio = l1_norm)
    for index in 1:size(X,1)
        single_distance_result[index] = distance(X[index,:], orig_entity, features_array, feature_distance_abs; norm_ratio = l1_norm)
        single_distance_result[index] = floor(single_distance_result[index]*10000000000)/10000000000.0
        combined_distance_result[index] = floor(combined_distance_result[index]*10000000000)/10000000000.0
        # if single_distance_result[index] != combined_distance_result[index]
        #     println("$index: $(single_distance_result[index]) with $(combined_distance_result[index])" )
        # end
    end
    @test combined_distance_result== single_distance_result
    # for combined norm
    combined_distance_result = distance(X, orig_entity, features; norm_ratio = combined_norm)
    for index in 1:size(X,1)
        single_distance_result[index] = distance(X[index,:], orig_entity, features_array, feature_distance_abs; norm_ratio = combined_norm)
        single_distance_result[index] = floor(single_distance_result[index]*10000000000)/10000000000.0
        combined_distance_result[index] = floor(combined_distance_result[index]*10000000000)/10000000000.0
        # if single_distance_result[index] != combined_distance_result[index]
        #     println("$index: $(single_distance_result[index]) with $(combined_distance_result[index])" )
        # end
    end

    # add the group distance
end;

# each feature in exactly one of groups

@testset "Feasible Test" begin
    # for the credit data set, check whether the results follow the constraints
    include("credit/credit_setup.jl");
    orig_instance = orig_entity
    explanation,  = @time explain(orig_instance, X, path, classifier)
    for i in 1:size(explanation,1)
        ## for each entity check whether "isMale" and "isMarried" not changed and "HasHistoryOfOverduePayments" only INCREASING
        @test (explanation[i, "isMale"]  == orig_instance["isMale"]) & (explanation[i, "isMarried"]  == orig_instance["isMarried"]) & (explanation[i, "HasHistoryOfOverduePayments"]  >= orig_instance["HasHistoryOfOverduePayments"])
    end

    # test on the feasible space compute

    # with delta representation (compress data)

    # manually call the init => check feasible and only one feature group changed
end