
using Pkg; Pkg.activate(".")
using GeCo, Test
include("../scripts/credit/credit_setup_MACE.jl");

@testset "Distance Test" begin
    # the set up for the distance test for credit dataset
    
    feasible_space = feasibleSpace(X, orig_instance, p)


    test_entity = DataFrame(orig_instance)
    l0_norm = [1.0, 0.0, 0.0, 0.0]
    l1_norm = [0.0, 1.0, 0.0, 0.0]
    l2_norm = [0.0, 0.0, 1.0, 0.0]
    combined_norm = [0.25, 0.25, 0.25, 0.25]

    # simple test that the distance of the same entity shuold be 0
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm) == [0.0]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm) == 0.0

    ## simple test of change one categorical feature for both distance functions 
    test_entity[1, "isMale"] = 1 - test_entity[1, "isMale"]
    #for l0 norm
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l0_norm) == [1.0/feasible_space.num_features]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l0_norm) == 1.0/feasible_space.num_features
    # for l1 norm
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l1_norm) == [1.0/feasible_space.num_features]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l1_norm) == 1.0/feasible_space.num_features
    # for l2 norm
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l2_norm) == [sqrt(1.0/feasible_space.num_features)]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l2_norm) == sqrt(1.0/feasible_space.num_features)
    #for combined norm
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm) == [1.0/feasible_space.num_features/2 + sqrt(1.0/feasible_space.num_features)/4+1/4]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm) == 1.0/feasible_space.num_features/2 + sqrt(1.0/feasible_space.num_features)/4+1/4
    test_entity[1, "isMale"] = 1 - test_entity[1, "isMale"]

    # ## simple test of change one Continuous feature for both distance functions 
    test_entity[1, "MaxBillAmountOverLast6Months"] += 10
    #for l0 norm
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l0_norm) == [1/feasible_space.num_features]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l0_norm) == 1/feasible_space.num_features
     # for l1 norm
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l1_norm) == [10.0/50810/feasible_space.num_features]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l1_norm) == 10.0/50810/feasible_space.num_features
    #for combined norm
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm) == [1.0/feasible_space.num_features/4 + 10.0/50810/feasible_space.num_features/4+10.0/50810/4 + sqrt(10.0/50810*10.0/50810/feasible_space.num_features)/4 ]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm) == 1.0/feasible_space.num_features/4 + 10.0/50810/feasible_space.num_features/4+10.0/50810/4 + sqrt(10.0/50810*10.0/50810/feasible_space.num_features)/4
    test_entity[1, "MaxBillAmountOverLast6Months"] -= 10

    ### simple test of changing several categorical (2) and Continuous (2) features
    test_entity[1, "isMale"] = 1 - test_entity[1, "isMale"]
    test_entity[1, "isMarried"] = 1 - test_entity[1, "isMarried"]
    test_entity[1, "MaxBillAmountOverLast6Months"] += 10
    test_entity[1, "EducationLevel"] += 2
    #for l0 norm
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l0_norm) == [4/feasible_space.num_features]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l0_norm) == 4/feasible_space.num_features
    # for l1 norm
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l1_norm) == [(2+2/3+10.0/50810)/feasible_space.num_features]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l1_norm) == (2+2/3+10.0/50810)/feasible_space.num_features
    #for combined norm
    @test distance(test_entity, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm) == [4.0/feasible_space.num_features/4 + (2+2/3+10.0/50810)/feasible_space.num_features/4 + sqrt((2+2*2/3/3+10.0/50810*10.0/50810)/feasible_space.num_features)/4 + 1.0/4]
    @test distance(test_entity[1,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm) == 4.0/feasible_space.num_features/4 + (2+2/3+10.0/50810)/feasible_space.num_features/4 + sqrt((2+2*2/3/3+10.0/50810*10.0/50810)/feasible_space.num_features)/4 + 1.0/4


    ### large tests mainly check whether the two measures return the same result

    ## check for all entities compare with the first entity (labeled as good)
    #for l1 norm
    single_distance_result = Array{Float64,1}(undef, size(X,1))
    combined_distance_result = distance(X, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l1_norm)
    for index in 1:size(X,1)
        single_distance_result[index] = distance(X[index,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l1_norm)
        single_distance_result[index] = floor(single_distance_result[index]*10000000000)/10000000000.0
        combined_distance_result[index] = floor(combined_distance_result[index]*10000000000)/10000000000.0
        # if single_distance_result[index] != combined_distance_result[index]
        #     println("$index: $(single_distance_result[index]) with $(combined_distance_result[index])" )
        # end
    end
    @test combined_distance_result== single_distance_result
    # for combined norm
    combined_distance_result = distance(X, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm)
    for index in 1:size(X,1)
        single_distance_result[index] = distance(X[index,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm)
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
    combined_distance_result = distance(X, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l1_norm)
    for index in 1:size(X,1)
        single_distance_result[index] = distance(X[index,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = l1_norm)
        single_distance_result[index] = floor(single_distance_result[index]*10000000000)/10000000000.0
        combined_distance_result[index] = floor(combined_distance_result[index]*10000000000)/10000000000.0
        # if single_distance_result[index] != combined_distance_result[index]
        #     println("$index: $(single_distance_result[index]) with $(combined_distance_result[index])" )
        # end
    end
    @test combined_distance_result== single_distance_result
    # for combined norm
    combined_distance_result = distance(X, orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm)
    for index in 1:size(X,1)
        single_distance_result[index] = distance(X[index,:], orig_instance, feasible_space.num_features, feasible_space.ranges; norm_ratio = combined_norm)
        single_distance_result[index] = floor(single_distance_result[index]*10000000000)/10000000000.0
        combined_distance_result[index] = floor(combined_distance_result[index]*10000000000)/10000000000.0
        # if single_distance_result[index] != combined_distance_result[index]
        #     println("$index: $(single_distance_result[index]) with $(combined_distance_result[index])" )
        # end
    end
    @test combined_distance_result== single_distance_result
    
end;

# each feature in exactly one of groups
@testset "feature group check" begin
    feasible_space = feasibleSpace(X, orig_instance, p)
    checker = false
    features = Array{Symbol, 1}(undef, feasible_space.num_features)
    for i in 1:14
        features[i] = :None
    end
    index = 1
    for feature_group in feasible_space.groups
        for feature in feature_group.names
            check = checker  || (feature in features)
            
            features[index] = feature
            index += 1
        end
    end
    @test checker == false
end

@testset "Feasible Test" begin
    # for the credit data set, check whether the results follow the constraints
    explanation,  = @time explain(orig_instance, X, p, classifier)
    for i in 1:size(explanation,1)
        ## for each entity check whether "isMale" and "isMarried" not changed and "HasHistoryOfOverduePayments" only INCREASING
        @test (explanation[i, "isMale"]  == orig_instance["isMale"]) & (explanation[i, "isMarried"]  == orig_instance["isMarried"]) & (explanation[i, "HasHistoryOfOverduePayments"]  >= orig_instance["HasHistoryOfOverduePayments"])
    end

    # test on the feasible space compute

    # with delta representation (compress data)

    # manually call the init => check feasible and only one feature group changed
end