struct DataManager
    orig_instance::DataFrameRow
    dict::Dict{BitVector, DataFrame}
    extended::Bool
end

initializeManager(orig_instance; extended=false) = begin
    DataManager(orig_instance, Dict{BitVector, DataFrame}(), extended)
end

get_store(manager::DataManager, mod::BitVector) = begin
    if haskey(manager.dict, mod)
        return manager.dict[mod]
    else
        orig_instance = manager.orig_instance
        keyList = [key for (i,key) in enumerate(keys(orig_instance)) if mod[i]]
        manager.dict[mod] = DataFrame([typeof(orig_instance[key])[] for key in keyList], keyList)

        # We extend the df with the extra columns for the genetic algorithm
        manager.extended && insertcols!(manager.dict[mod], :score=>Float64[], :outc=>Bool[], :estcf=>Bool[])

        return manager.dict[mod]
    end
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
