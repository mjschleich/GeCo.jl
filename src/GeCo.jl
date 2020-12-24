module GeCo

export explain, actions, featureList, initializeFeatures, testExplanations

using DataFrames, Statistics, PyCall
import ScikitLearn, JSON, StatsBase, MLJ

const default_norm_ratio = [0.25, 0.25, 0.25, 0.25]

const NUM_EXTRA_COL = 4
const extra_col = [:score, :outc, :estcf, :mod]
const NUM_EXTRA_FEASIBLE_SPACE_COL = 2

# Implements the struct for features and feature groups, as well as the methods to initializeFeatures
# include("utils/FeatureStruct.jl")
# export FeatureGroup

# Implements the struct for features and feature groups, as well as the methods to initializeFeatures
include("components/plaf.jl")
export @PLAF, @GROUP, PLAFProgram, initPLAF

# Implements the struct for features and feature groups, as well as the methods to initializeFeatures
include("components/feasibleSpace.jl")
export feasibleSpace, initDomains, ground, FeatureGroup

# Implementation of the Δ-representation
include("dataManager/DataManager.jl")
export materialize, DataManager

# Implements partial evaluation for Random Forest Classifiers
include("classifier/RandomForestEval.jl")
using .RandomForestEvaluation
export initRandomForestEval, initPartialRandomForestEval, predict

# Implements partial evaluation for multi-layered perceptrons
include("classifier/MLPEval.jl")
using .MLPEvaluation
export initMLPEval, predict

# Implementation of various score function, which overloads prediction functions for different ML packages
include("components/score.jl")
export score

# Implementation of the distance function used by GeCo
include("components/distance.jl")
export distance, distanceFeatureGroup, minimumObservableCounterfactual, observableCounterfactuals

# Implementation of mutation operator
include("components/initialPopulation.jl")
export initialPopulation

# Implementation of crossover operator
include("components/crossover.jl")
export crossover!

# Implementation of mutation operator
include("components/mutation.jl")
export mutation!

# Implementation of mutation operator
include("components/selection.jl")
export selection!

function explain(orig_instance::DataFrameRow, data::DataFrame, program::PLAFProgram, classifier;
    desired_class=1,
    k::Int64=100,
    max_num_generations::Int64=100,
    min_num_generations::Int64=3,
    max_num_samples::Int64=5,
    convergence_k::Int=5,
    norm_ratio::Array{Float64,1}=default_norm_ratio,
    domains::Vector{DataFrame}=Vector{DataFrame}(),
    compress_data::Bool=false,
    return_df::Bool=false,
    ablation::Bool=false,
    run_crossover::Bool=true,
    run_mutation::Bool=true,
    verbose::Bool=false
    )

    # Compute the feasible space for each feature group
    feasible_space = feasibleSpace(data, orig_instance, program; domains=domains)

    distance_temp = Array{Float64, 1}(undef, 100000)
    representation_size = zeros(Int64, max_num_generations+1)

    generation::Int64 = 0
    count::Int64 = 0
    converged::Bool = false

    if !ablation && verbose

        print("-- Time init pop:\t")
        population = @time initialPopulation(orig_instance, feasible_space; compress_data=compress_data)

        count += size(population,1)
        representation_size[1] = (compress_data ? size(population,2) : nrow(population) * ncol(population))

        print("-- Time selection:\t")
        @time selection!(population, k, orig_instance, feasible_space, classifier, desired_class;
            norm_ratio=norm_ratio,
            distance_temp=distance_temp)

        @time while generation < min_num_generations || !converged && generation < max_num_generations
            println("Generation: ", generation+1)

            print("-- Time Crossover:\t")
            @time crossover!(population, orig_instance, feasible_space)

            print("-- Time Mutation:\t")
            @time mutation!(population, feasible_space; max_num_samples = max_num_samples)

            count::Int64 += max(0, size(population,1) - k)
            representation_size[generation+1] = (compress_data ? size(population,2) : nrow(population) * ncol(population))

            print("-- Time Selection:\t")
            converged = @time selection!(population, k, orig_instance, feasible_space, classifier, desired_class;
                norm_ratio=norm_ratio, distance_temp=distance_temp, convergence_k=convergence_k)

            generation += 1
        end

        println(
            "Number of generated counterfactuals: $count\n" *
            "Number of generations:               $(generation)")

    elseif !ablation

        population = initialPopulation(orig_instance, feasible_space; compress_data=compress_data)

        count += size(population,1)
        representation_size[1] = (compress_data ? size(population,2) : nrow(population) * ncol(population))

        selection!(population, k, orig_instance, feasible_space, classifier, desired_class;
            norm_ratio=norm_ratio,
            distance_temp=distance_temp)

        while generation < min_num_generations || !converged && generation < max_num_generations

            crossover!(population, orig_instance, feasible_space)

            mutation!(population, feasible_space; max_num_samples = max_num_samples)

            size_pop = size(population)
            count += max(0, size_pop[1] - k)
            representation_size[generation+1] = (compress_data ? size_pop[2] : size_pop[1] * size_pop[2])

            converged = selection!(population, k, orig_instance, feasible_space, classifier, desired_class;
                norm_ratio=norm_ratio, distance_temp=distance_temp, convergence_k=convergence_k)

            generation += 1
        end
    else
        selection_time = 0.0
        mutation_time = 0.0
        crossover_time = 0.0

        prep_time = @elapsed (population) = initialPopulation(orig_instance, feasible_space; compress_data=compress_data)

        count += size(population,1)
        representation_size[1] = (compress_data ? size(population,2) : nrow(population) * ncol(population))

        stime = @elapsed selection!(population, k, orig_instance, feasible_space, classifier, desired_class;
            norm_ratio=norm_ratio,
            distance_temp=distance_temp)

        prep_time += stime

        while generation < min_num_generations || !converged && generation < max_num_generations

            if run_crossover
                ctime = @elapsed crossover!(population, orig_instance, feasible_space)
                crossover_time += ctime
            end

            if run_mutation
                mtime = @elapsed mutation!(population, feasible_space; max_num_samples = max_num_samples)
                mutation_time += mtime
            end

            size_pop = size(population)
            count += max(0, size_pop[1] - k)
            representation_size[generation+1] = (compress_data ? size_pop[2] : size_pop[1] * size_pop[2])

            stime = @elapsed (converged) = selection!(population, k, orig_instance, feasible_space, classifier, desired_class;
                norm_ratio=norm_ratio, distance_temp=distance_temp, convergence_k=convergence_k)
            selection_time += stime

            generation += 1
        end

        if return_df && compress_data
            population = materialize(population)
            sort!(population, :score)
        end

        return population, count, generation, representation_size, prep_time, selection_time, mutation_time, crossover_time
    end

    if return_df && compress_data
        population = materialize(population)
        sort!(population, :score)
    end

    return population, count, generation, representation_size
end


function actions(counterfactuals::DataFrame, orig_instance; num_actions = 5)
    for idx in 1:num_actions
        cf = counterfactuals[idx,:]
        println("\n------- COUNTERFACTUAL $idx\nDesired Outcome: $(cf.outc),\tScore: $(cf.score)")
        for feature in propertynames(orig_instance)
            delta = cf[feature] - orig_instance[feature]
            if delta != 0
                println(feature, " : \t",orig_instance[feature], " => ", cf[feature])
            end
        end
    end
end


function actions(counterfactuals::DataManager, orig_instance; num_actions = 5)

    # Turn DataManager into a DataFrame
    df = materialize(counterfactuals)
    sort!(df, :score)

    for idx in 1:num_actions
        cf = df[idx,:]
        println("\n------- COUNTERFACTUAL $idx\nDesired Outcome: $(cf.outc),\tScore: $(cf.score)")
        for feature in String.(names(orig_instance))
            delta = cf[feature] - orig_instance[feature]
            if delta != 0
                println(feature, " : \t",orig_instance[feature], " => ", cf[feature])
            end
        end
    end
end

function testExplanations(explanations, orig_instance)
    for (row, explanation) in enumerate(eachrow(explanations))

        for (i,f) in enumerate(eachindex(explanation[1:end-3]))
            if explanation.mod[i] && explanation[f] == orig_instance[f]
                println("In Row $row: $(explanation[f]) == $(orig_instance[f]) for $f")
            end
        end
    end
end

end
