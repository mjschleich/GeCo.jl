mutable struct DataManager
    orig_instance::DataFrameRow
    dict::Dict{BitVector, DataFrame}
    extended::Bool
end

initializeManager(orig_instance; extended=false) = begin
    DataManager(orig_instance, Dict{BitVector, DataFrame}(), extended)
end

get_store_impl(dict::Dict{BitVector, DataFrame}, mod::BitVector, orig_instance, extended) = begin
    if haskey(dict, mod)
        return dict[mod]
    else
        keyList = [key for (i,key) in enumerate(keys(orig_instance)) if mod[i]]
        dict[mod] = DataFrame([typeof(orig_instance[key])[] for key in keyList], keyList)

        # We extend the df with the extra columns for the genetic algorithm
        extended && insertcols!(dict[mod], :score=>Float64[], :outc=>Bool[], :estcf=>Bool[])

        return dict[mod]
    end
end

get_store(manager::DataManager, mod::BitVector) = begin
    get_store_impl(manager.dict, mod, manager.orig_instance, manager.extended)
end

function Base.empty!(manager::DataManager)
    empty!(manager.dict)
end

function Base.push!(manager::DataManager, mod::BitVector, partial_entity)
    entities = get_store(manager, mod)
    push!(entities, partial_entity; cols=:subset)
end

function Base.append!(manager::DataManager, mod::BitVector, partial_entities)
    entities = get_store(manager, mod)
    append!(entities, partial_entities, cols=:subset)
end

function Base.keys(manager::DataManager)
    return keys(manager.dict)
end

Base.size(manager::DataManager) = Tuple(sum([nrow(df), nrow(df)*ncol(df)] for df in values(manager.dict)))
function Base.size(manager::DataManager, i::Integer)
    if i == 1
        size(manager)[1]
    elseif i == 2
        size(manager)[2]
    else
        throw(ArgumentError("DataManager only has two dimensions"))
    end
end


function Base.delete!(manager, mod)
    delete!(manager.dict, mod)
end

function select!(manager, mod, keeps::BitVector)
    manager.dict[mod] = manager.dict[mod][keeps,:]
end

# Turns the DataManager into a DataFrame
function materialize(manager::DataManager)::DataFrame

    num_rows= size(manager,1)
    df = repeat(DataFrame(manager.orig_instance), num_rows)

    insertcols!(df,
        :score=>zeros(Float64, num_rows),
        :outc=>falses(num_rows),
        :estcf=>falses(num_rows),
        :mod=>Vector{BitArray}(undef, num_rows)
        )

    i = 1

    cols = falses(size(df,2))
    for (mod, entities) in manager.dict

        for i in 1:length(mod)
            cols[i] = mod[i]
        end

        df[i:i+nrow(entities)-1, cols] = entities[1:end-3]
        # println(mod, df[i:i+nrow(entities)-1, cols], entities)
        df[i:i+nrow(entities)-1, :estcf] = entities[:, :estcf]
        df[i:i+nrow(entities)-1, :score] = entities[:, :score]
        df[i:i+nrow(entities)-1, :outc] = entities[:, :outc]
        df[i:i+nrow(entities)-1, :mod] = (mod for _=i:i+nrow(entities)-1)
        i += nrow(entities)
    end

    return df
end

function Base.show(io::IO, manager::DataManager)
    print(io, "Data Manager (Groups=$(length(manager.dict)),  Entities=$(size(manager,1)))")
end

struct DeltaDataFrameWrapper
    instance:: DataFrameRow
    orig_instance:: DataFrameRow
end

function Base.getproperty(wrapper::DeltaDataFrameWrapper, property::Symbol)
    if haskey(getfield(wrapper, :instance), property)
        getproperty(getfield(wrapper, :instance), property)
    else
        getproperty(getfield(wrapper, :orig_instance), property)
    end
end


function actionCascade(manager::DataManager, implications::Vector{GroundedImplication})
    bv = BitVector(undef, length(manager.orig_instance))
    # max_nrows = maximum(nrows(df) for (mod, df) in manager.dict)

    dict = manager.dict
    for impl in implications

        affect_bits = impl.condFeatures .| impl.conseqFeaturesBitVec
        to_add = Dict{BitVector, DataFrame}()

        for (mod, df) in dict

            # continue if this is unrelated
            bv .= mod .& affect_bits
            if !any(bv)
                df_to_add = get_store_impl(to_add, mod, manager.orig_instance, manager.extended)
                append!(df_to_add, df)
                continue
            end

            # calculate validInstances
            validInstances = .!impl.condition.(DeltaDataFrameWrapper(instance, manager.orig_instance) for instance in eachrow(df))

            # store valid instances for next round
            if count(validInstances) != 0
                df_to_add = get_store_impl(to_add, mod, manager.orig_instance, manager.extended)
                append!(df_to_add, df[validInstances, :])
            end

            # there are invalid instances and we should process them and store for the next round
            if count(validInstances) != length(validInstances) && !isempty(impl.sampleSpace)
            
                # println("There are $(count(.!validInstances)) invalid cases, below are one example")
                # println(df[.!validInstances, :][1, :])
                # println()
                
                # tweak and add previously invalid instances
                refined_mod = copy(mod)
                refined_mod .|= impl.conseqFeaturesBitVec
                df_to_add = get_store_impl(to_add, refined_mod, manager.orig_instance, manager.extended)

                fspace = impl.sampleSpace
                features = impl.conseqFeatures
                num_tuples = length(validInstances) - count(validInstances)
                sampled_rows = StatsBase.sample(1:nrow(fspace), StatsBase.FrequencyWeights(fspace.count), num_tuples)

                # during hcat, DataFrame will check that the columns from different dfs should be different, so
                # we need to exclude common columns first

                original_keys = keys(manager.orig_instance)[(mod .⊻ impl.conseqFeaturesBitVec) .& mod]
                manager.extended && push!(original_keys, :score, :outc, :estcf)

                append!(df_to_add, hcat(
                    df[.!validInstances, original_keys],
                    fspace[sampled_rows, features]))
            end
        end

        dict = to_add
    end

    manager.dict = dict
end
