using GeneralizedGenerated

struct FeatureGroup
    features::Tuple{Vararg{Symbol}}
    names::Vector{Symbol}
    indexes::BitVector
    allCategorical::Bool
end

struct FeasibleSpace
    groups::Vector{FeatureGroup}
    ranges::Dict{Symbol,Float64}
    num_features::Int64
    feasibleSpace::Vector{DataFrame}
end

function addkey!(membernames, nam)::Symbol
    if !haskey(membernames, nam)
        membernames[nam] = gensym()
    end
    membernames[nam]
end

onearg(e, f) = e.head == :call && length(e.args) == 2 && e.args[1] == f
mapexpr(f, e) = Expr(e.head, map(f, e.args)...)

# function replace_dotted!(e, membernames, orig_instance)
#     if e.args[1] ∈ (QuoteNode(:cf), QuoteNode(:x_cf), QuoteNode(:counterfactual))
#         return replace_syms!(e.args[2], membernames, orig_instance)
#     elseif e.args[1] ∈ (QuoteNode(:x), QuoteNode(:inst), QuoteNode(:instance))
#         return orig_instance[e.args[2].value]
#     else
#         @error "The tuple identifier should be one of (cf, x_cf, counterfactual) or (x, inst, instance), we got $(e.args[1])"
#     end
# end

####
# TODO: FIXME: Add support for IF statements
####
replace_syms!(x, membernames, orig_instance) = x
replace_syms!(q::QuoteNode, membernames, orig_instance) = replace_syms!(Meta.quot(q.value), membernames, orig_instance)

function replace_syms!(e::Expr, membernames, orig_instance)
    # if onearg(e, :^)
    #     e.args[2]
    # elseif onearg(e, :cols)
    #     addkey!(membernames, :($(e.args[2])))
    if e.head == :quote
        addkey!(membernames, Meta.quot(e.args[1]) )
    elseif e.head == :.
        if e.args[1] ∈ (QuoteNode(:cf), QuoteNode(:x_cf), QuoteNode(:counterfactual))
            return replace_syms!(e.args[2], membernames, orig_instance)
        elseif e.args[1] ∈ (QuoteNode(:x), QuoteNode(:inst), QuoteNode(:instance))
            return orig_instance[e.args[2].value]
        else
            @error "The tuple identifier should be one of (cf, x_cf, counterfactual) or (x, inst, instance), we got $(e.args[1])"
        end
    else
        e2 = mapexpr(x -> replace_syms!(x, membernames, orig_instance), e)
    end
end

function make_source_concrete(x::AbstractVector)
    if isempty(x) || isconcretetype(eltype(x))
        return x
    elseif all(t -> t isa Union{AbstractString, Symbol}, x)
        return Symbol.(x)
    else
        throw(ArgumentError("Column references must be either all the same " *
                            "type or a a combination of `Symbol`s and strings"))
    end
end

function ground(constraints::Vector{Expr}, orig_instance::DataFrameRow)
    grounded_constraints = Pair{Vector{Symbol}, Any}[]

    for kw in constraints
        membernames = Dict{Any, Symbol}()

        body::Expr = replace_syms!(kw, membernames, orig_instance)
        source::Expr = Expr(:vect, keys(membernames)...)
        inputargs::Expr = Expr(:tuple, values(membernames)...)

        generated_func = quote
            GeCo.make_source_concrete($(source)) => $inputargs -> $body
        end

        push!(grounded_constraints, runtime_eval(generated_func))
    end

    return grounded_constraints
end


function initGroups(prog::PLAFProgram, data::DataFrame)

    feature_list = Array{Feature,1}()
    groups = Array{FeatureGroup, 1}()

    addedFeatures = Set{Symbol}()
    onehotFeature = falses(ncol(data))

    for group in prog.groups

        allCategorical = true

        indexes = falses(ncol(data))

        for (idx, feature) in enumerate(group)
            @assert feature ∈ propertynames(data) "The feature $feature does not occur in the input data."
            @assert feature ∉ addedFeatures "Each feature can be in at most one group!"
            push!(addedFeatures, feature)
            indexes[findfirst(isequal(feature),propertynames(data))] = true
            allCategorical = (elscitype(data[!,feature]) == Multiclass)
        end

        push!(groups,
            FeatureGroup(
                group,
                [group...],
                indexes,
                allCategorical
                )
            )
    end

    for feature in propertynames(data)
        if feature ∉ addedFeatures
            @assert feature ∈ propertynames(data) "The feature $feature does not occur in the input data."

            indexes = falses(ncol(data))
            indexes[findfirst(isequal(feature),propertynames(data))] = true

            push!(groups,
                FeatureGroup(
                    (feature,),
                    [feature],
                    indexes,
                    elscitype(data[!,feature]) == Multiclass
                    )
            )
        end
    end

    return groups
end

function initDomains(prog::PLAFProgram, data::DataFrame)::Vector{DataFrame}
    groups = initGroups(prog, data)
    return initDomains(groups, data)
end

function initDomains(groups::Vector{FeatureGroup}, data::DataFrame)::Vector{DataFrame}
    domains = Vector{DataFrame}(undef, length(groups))
    for (gidx, group) in enumerate(groups)
        feat_syms = group.features
        if length(feat_syms) > 1000
            domains[gidx] = data[!, feat_syms]
            domains[gidx].count = ones(Int64, nrow(data))
        else
            domains[gidx] = combine(groupby(data[!,[feat_syms...]], [feat_syms...]), nrow => :count)
        end
    end
    return domains
end



function feasibleSpace(data::DataFrame, orig_instance::DataFrameRow, prog::PLAFProgram;
    domains::Vector{DataFrame}=Vector{DataFrame}())::FeasibleSpace

    constraints = ground(prog.constraints,orig_instance)

    groups = initGroups(prog,data)
    ranges = Dict(feature => Float64(maximum(col)-minimum(col)) for (feature, col) in pairs(eachcol(data)))
    num_features = ncol(data)

    feasible_space = Vector{DataFrame}(undef, length(groups))

    if isempty(domains)
        domains = initDomains(groups, data)
    else
        @assert length(domains) == length(groups)
    end

    satisfied_constraints = falses(length(constraints))

    for (gidx, group) in enumerate(groups)

        features = [group.features...]
        fspace = filter(row -> row[features] != orig_instance[features], domains[gidx])

        for (cidx, constraint) in enumerate(constraints)
            if constraint[1] ⊆ group.features
                @assert !satisfied_constraints[cidx]

                # Compute constraints on this group
                filter!(constraint, fspace)
                satisfied_constraints[cidx] = true
            end
        end

        fspace.distance = distance(fspace, orig_instance, num_features, ranges)
        feasible_space[gidx] = fspace
    end

    for (cidx, constraint) in enumerate(constraints)
        if !satisfied_constraints[cidx]
            println("We have an extra constraint to columns: $(constraint[1])")
        end
    end

    return FeasibleSpace(groups, ranges, num_features, feasible_space)
end





# function feasibleSpace2(orig_instance::DataFrameRow, groups::Vector{FeatureGroup}, domains::Vector{DataFrame})::Vector{DataFrame}
#     # We create a dictionary of constraints for each feature
#     # A constraint is a function
#     feasibility_constraints = Dict{String,Function}()
#     feasible_space = Vector{DataFrame}(undef, length(groups))

#     @assert length(domains) == length(groups)

#     # This should come from the constraint parser
#     # actionable_features = []

#     for (gidx, group) in enumerate(groups)

#         domain = domains[gidx]

#         feasible_rows = trues(nrow(domain))
#         identical_rows = trues(nrow(domain))

#         for feature in group.features

#             orig_val::Float64 = orig_instance[feature.name]
#             col::Vector{Float64} = domain[!, feature.name]

#             identical_rows .&= (col .== orig_val)

#             if !feature.actionable
#                 feasible_rows .&= (col .== orig_val)
#             elseif feature.constraint == INCREASING
#                 feasible_rows .&= (col .> orig_val)
#             elseif feature.constraint == DECREASING
#                 feasible_rows .&= (col .< orig_val)
#             end
#         end

#         # Remove the rows that have identical values - we don't want to sample the original instance
#         feasible_rows .&= .!identical_rows

#         feasible_space[gidx] = domains[gidx][feasible_rows, :]
#         feasible_space[gidx].distance = distanceFeatureGroup(feasible_space[gidx], orig_instance, group)
#     end

#     return feasible_space
# end

# function feasibleSpace2(orig_instance::DataFrameRow, groups::Vector{FeatureGroup}, data::DataFrame)::Vector{DataFrame}
#     # We create a dictionary of constraints for each feature
#     # A constraint is a function
#     feasibility_constraints = Dict{String,Function}()
#     feasible_space = Vector{DataFrame}(undef, length(groups))

#     # This should come from the constraint parser
#     actionable_features = []

#     feasible_rows = trues(size(data,1))
#     identical_rows = trues(size(data,1))

#     for (gidx, group) in enumerate(groups)

#         fill!(feasible_rows, true)
#         fill!(identical_rows, true)

#         feat_names = String[]
#         for feature in group.features
#             identical_rows .&= (data[:, feature.name] .== orig_instance[feature.name])

#             if !feature.actionable
#                 feasible_rows .&= (data[:, feature.name] .== orig_instance[feature.name])
#             elseif feature.constraint == INCREASING
#                 feasible_rows .&= (data[:, feature.name] .> orig_instance[feature.name])
#             elseif feature.constraint == DECREASING
#                 feasible_rows .&= (data[:, feature.name] .< orig_instance[feature.name])
#             end
#             push!(feat_names, feature.name)
#         end

#         # Remove the rows that have identical values - we don't want to sample the original instance
#         feasible_rows .&= .!identical_rows

#         feasible_space[gidx] = combine(
#                 groupby(
#                     data[feasible_rows, feat_names],
#                     feat_names),
#                 nrow => :count
#                 )

#         feasible_space[gidx].distance = distanceFeatureGroup(feasible_space[gidx], orig_instance, group)
#     end

#     return feasible_space
# end
