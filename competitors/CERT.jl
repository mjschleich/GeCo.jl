using DataFrames, Statistics
import StatsBase
import JSON
import MLJ
import GeCo: FeasibleSpace

function initialPopulation_naive(data::DataFrame, orig_instance::DataFrameRow, classifier, desired_class::Int64, program::PLAFProgram)

    preds = ScikitLearn.predict_proba(classifier, MLJ.matrix(data))[:, desired_class+1]

    outcome = (preds .> 0.5)

    return observableCounterfactuals(data, outcome, orig_instance, program.constraints, program.implications;
            check_feasibility=true, desired_class=desired_class)
end

function crossover_naive!(population::DataFrame, feature_groups::Vector{FeatureGroup}; crossover_prob::Float64=0.5)

    # A portion of individuals (dependent on crossover_prob) are subjected to crossver\
    # Involves randomly interchanging some features values between individuals
    row_num = floor(Int, nrow(population)*crossover_prob)
    parents  = StatsBase.sample(1:nrow(population), row_num; replace=false, ordered=true)
    for index in 1:floor(Int,row_num/2)
        parent_1 = population[parents[index], :]
        parent_2 = population[parents[index+1], :]
        push!(population,parent_1)
        push!(population,parent_2)

        # changed_num = StatsBase.sample(1:size(population)[2]-1, 1)[1]
        # sample the groups to crossover
        changed_num = StatsBase.sample(1:length(feature_groups), 1, replace=false)[1]
        changed_groups = StatsBase.sample(1:length(feature_groups), changed_num, replace=false)

        for changed_group in changed_groups
            for changed_feature in feature_groups[changed_group].names
                tmp = parent_1[changed_feature]
                parent_1[changed_feature] = parent_2[changed_feature]
                parent_2[changed_feature] = tmp
            end
        end
    end
end

function mutation_naive!(population::DataFrame, data::DataFrame, feasible_space; mutation_prob::Float64=0.5)

    # Select portion of counterfactuals (according to probablity p_m) to arbitrarly change some features
    # Randomly mutate some feature

    feature_groups::Vector{FeatureGroup} = feasible_space.groups
    # Add the new coutnerfactual to the population
    fnames = names(data)
    row_num = floor(Int, nrow(population)*mutation_prob)
    parents  = StatsBase.sample(1:nrow(population), row_num; replace=false, ordered=true)
    for index in 1:row_num
        original = population[parents[index], :]
        push!(population,original)

        # sample groups to mutate
        changed_num = StatsBase.sample(1:length(feature_groups))
        changed_groups = StatsBase.sample(1:length(feature_groups), changed_num, replace=false)

        for changed_group in changed_groups
            space = feasible_space.feasibleSpace[changed_group]
            isempty(space) && continue

            value = StatsBase.sample(1:nrow(space), StatsBase.FrequencyWeights(space.count))
            original[feature_groups[changed_group].names] = space[value, feature_groups[changed_group].names]
        end
    end

end

function selection_naive!(population::DataFrame, k::Int64, orig_instance::DataFrameRow, classifier, desired_class::Int64, feasible_space::FeasibleSpace;
    norm_ratio::Array{Float64,1}=default_norm_ratio,
    convergence_k::Int=10,
    distance_temp::Vector{Float64}=Vector{Float64}(undef,100))


    ranges = feasible_space.ranges
    num_features = feasible_space.num_features

    # This should be similar to the conventional selection function
    # But we should restrict the population to only the ones that get the desired outcome
    preds = ScikitLearn.predict_proba(classifier, MLJ.matrix(population[!, 1:end-1]))[:, desired_class+1]

    delete!(population, preds .< 0.5)

    population[!,:score] = distance(population, orig_instance, num_features, ranges;
        norm_ratio=norm_ratio, distance_temp=distance_temp)

    sort!(population, [:score])

    # We keep the top-k counterfactuals
    unique!(population)

    # println(size(population,1) > k)
    delete!(population, (k+1:size(population,1)))
end

function explain_naive(orig_instance::DataFrameRow, data::DataFrame, program::PLAFProgram, classifier;
    desired_class=1,
    k::Int64=100,
    num_generations::Int64=100,
    max_num_samples::Int64=5,
    norm_ratio::Array{Float64,1}=[0.25, 0.25, 0.25, 0.25],
    verbose::Bool=false
    )

    feasible_space = feasibleSpace(data, orig_instance, program)

    population = initialPopulation_naive(data, orig_instance, classifier, desired_class, program)

    if isempty(population)
        return nothing
    end

    population.score = zeros(nrow(population))

    distance_temp = Array{Float64, 1}(undef, max(nrow(population),10000)*4)

    selection_naive!(population, k, orig_instance, classifier, desired_class, feasible_space;
        norm_ratio=norm_ratio, distance_temp=distance_temp)

    for generation in 1:num_generations

        crossover_naive!(population, feasible_space.groups)

        mutation_naive!(population, data, feasible_space)

        selection_naive!(population, k, orig_instance, classifier, desired_class, feasible_space;
            norm_ratio=norm_ratio, distance_temp=distance_temp)

    end
    return population
end

