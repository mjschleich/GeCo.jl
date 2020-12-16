struct PLAFProgram
    groups::Vector{Tuple{Vararg{Symbol}}}
    constraints::Vector{Expr}
end

initPLAF() = PLAFProgram(Vector{Tuple}(), Vector{Expr}())

Base.empty!(p::PLAFProgram) = begin
    empty!(p.groups)
    empty!(p.constraints)
    p
end

plaf_helper(x,  t) =
begin
    quote
        $append!($x.constraints, $t)
    end
end

group_helper(x, t) =
begin
    quote
        $push!($x.groups, $t)
    end
end

macro PLAF(x, args...)
    esc(plaf_helper(x,args))
end

macro GROUP(x, args...)
    @assert typeof(args) <: Tuple{Symbol,Vararg{Symbol}} "A group should be a list of features not $(typeof(args))"
    esc(group_helper(x,args))
end

function addkey!(membernames, nam)
    if !haskey(membernames, nam)
        membernames[nam] = gensym()
    end
    membernames[nam]
end

onearg(e, f) = e.head == :call && length(e.args) == 2 && e.args[1] == f
mapexpr(f, e) = Expr(e.head, map(f, e.args)...)

function replace_dotted!(e, membernames, orig_instance)
    if e.args[1] === QuoteNode(:cf) ||  e.args[1] === QuoteNode(:x_cf) || e.args[1] === QuoteNode(:counterfactual)
        # println(e.args[1], " -- ", typeof(e.args[2]))
        return replace_syms!(e.args[2], membernames, orig_instance)
    elseif e.args[1] === QuoteNode(:x) ||  e.args[1] === QuoteNode(:inst) || e.args[1] === QuoteNode(:instance)
        # println(e.args[1])
        return orig_instance[e.args[2].value]
    else
        @error "The tuple identifier should be one of (cf, x_cf, counterfactual) or (x, inst, instance), we got $(e.args[1])"
    end
end

replace_syms!(x, membernames, orig_instance) = x
replace_syms!(q::QuoteNode, membernames, orig_instance) = replace_syms!(Meta.quot(q.value), membernames, orig_instance)
replace_syms!(e::Expr, membernames, orig_instance) =
    if onearg(e, :^)
        e.args[2]
    # elseif onearg(e, :_I_)
    #     @warn "_I_() for escaping variables is deprecated, use cols() instead"
    #     addkey!(membernames, :($(e.args[2])))
    elseif onearg(e, :cols)
        addkey!(membernames, :($(e.args[2])))
    elseif e.head == :quote
        addkey!(membernames, Meta.quot(e.args[1]) )
    elseif e.head == :.
        e2 = replace_dotted!(e, membernames, orig_instance)
    else
        e2 = mapexpr(x -> replace_syms!(x, membernames, orig_instance), e)
    end

function make_source_concrete(x::AbstractVector)
    # println(" make_source_concrete ")
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

    grounded_constraints = []

    for kw in constraints
        membernames = Dict{Any, Symbol}()

        # act on f(:x)
        body = replace_syms!(kw, membernames, orig_instance)

        println("body: $body membernames: ", membernames)

        source = Expr(:vect, keys(membernames)...)
        inputargs = Expr(:tuple, values(membernames)...)

        grounded_expr = quote
            GeCo.make_source_concrete($(source)) =>
            $inputargs -> begin
                $body
            end
        end

        push!(grounded_constraints, eval(grounded_expr))
    end

    return grounded_constraints
end


function initDomains(path::String, data::DataFrame)::Vector{DataFrame}
    _, groups = initializeFeatures(path*"/data_info.json", data)
    return initDomains(groups, data)
end


function initDomains(groups::Vector{FeatureGroup}, data::DataFrame)::Vector{DataFrame}

    feasible_space = Vector{DataFrame}(undef, length(groups))
    for (gidx, group) in enumerate(groups)
        feat_names = group.names
        feat_sym = Symbol.(feat_names)
        if length(feat_sym) > 1000
            feasible_space[gidx] = data[!, feat_sym]
            feasible_space[gidx].count = ones(Int64, nrow(data))
        else
            feasible_space[gidx] = combine(groupby( data[!, feat_sym], feat_sym), nrow => :count)
        end
    end
    return feasible_space
end

function feasibleSpace(orig_instance::DataFrameRow, groups::Vector{FeatureGroup}, domains::Vector{DataFrame})::Vector{DataFrame}
    # We create a dictionary of constraints for each feature
    # A constraint is a function
    feasibility_constraints = Dict{String,Function}()
    feasible_space = Vector{DataFrame}(undef, length(groups))

    @assert length(domains) == length(groups)

    # This should come from the constraint parser
    # actionable_features = []

    for (gidx, group) in enumerate(groups)

        domain = domains[gidx]

        feasible_rows = trues(nrow(domain))
        identical_rows = trues(nrow(domain))

        for feature in group.features

            orig_val::Float64 = orig_instance[feature.name]
            col::Vector{Float64} = domain[!, feature.name]

            identical_rows .&= (col .== orig_val)

            if !feature.actionable
                feasible_rows .&= (col .== orig_val)
            elseif feature.constraint == INCREASING
                feasible_rows .&= (col .> orig_val)
            elseif feature.constraint == DECREASING
                feasible_rows .&= (col .< orig_val)
            end
        end

        # Remove the rows that have identical values - we don't want to sample the original instance
        feasible_rows .&= .!identical_rows

        feasible_space[gidx] = domains[gidx][feasible_rows, :]
        feasible_space[gidx].distance = distanceFeatureGroup(feasible_space[gidx], orig_instance, group)
    end

    return feasible_space
end


function feasibleSpace2(orig_instance::DataFrameRow, groups::Vector{FeatureGroup}, domains::Vector{DataFrame}, constraints)::Vector{DataFrame}

    feasible_space = Vector{DataFrame}(undef, length(groups))

    @assert length(domains) == length(groups)

    for (gidx, group) in enumerate(groups)

        domain = domains[gidx]

        fspace = filter([group...] => (orig_instance[feature.name] .!= domain[!, feature.name]), domain)

        for (cidx, constraint) in enumerate(constraints)
            if constraint[1] ⊆ group
                # Compute constraints on this group
                filter!(constraint, fspace)

                # TODO: Keep track of computed constraints (with a bitset)
            end
        end

        fspace.distance = distanceFeatureGroup(fspace, orig_instance, group)
        feasible_space[gidx] = fspace

        # Remove the rows that have identical values - we don't want to sample the original instance
        # feasible_rows .&= .!identical_rows
        # feasible_space[gidx] = domains[gidx][feasible_rows, :]
        # feasible_space[gidx].distance = distanceFeatureGroup(feasible_space[gidx], orig_instance, group)
        # for feature in group.features
        #     orig_val::Float64 = orig_instance[feature.name]
        #     col::Vector{Float64} = domain[!, feature.name]
        #     identical_rows .&= (col .== orig_val)
        #     if !feature.actionable
        #         feasible_rows .&= (col .== orig_val)
        #     elseif feature.constraint == INCREASING
        #         # expr = :($col .> $orig_val)
        #         # println(expr_increase, col, orig_val)
        #         # println(eval(expr_increase))
        #         # feasible_rows .&= eval(expr_increase)
        #     elseif feature.constraint == DECREASING
        #         feasible_rows .&= (col .< orig_val)
        #     end
        # end
    end

    return feasible_space
end


function feasibleSpace(orig_instance::DataFrameRow, groups::Vector{FeatureGroup}, data::DataFrame)::Vector{DataFrame}
    # We create a dictionary of constraints for each feature
    # A constraint is a function
    feasibility_constraints = Dict{String,Function}()
    feasible_space = Vector{DataFrame}(undef, length(groups))

    # This should come from the constraint parser
    actionable_features = []

    feasible_rows = trues(size(data,1))
    identical_rows = trues(size(data,1))

    for (gidx, group) in enumerate(groups)

        fill!(feasible_rows, true)
        fill!(identical_rows, true)

        feat_names = String[]
        for feature in group.features
            identical_rows .&= (data[:, feature.name] .== orig_instance[feature.name])

            if !feature.actionable
                feasible_rows .&= (data[:, feature.name] .== orig_instance[feature.name])
            elseif feature.constraint == INCREASING
                feasible_rows .&= (data[:, feature.name] .> orig_instance[feature.name])
            elseif feature.constraint == DECREASING
                feasible_rows .&= (data[:, feature.name] .< orig_instance[feature.name])
            end
            push!(feat_names, feature.name)
        end

        # Remove the rows that have identical values - we don't want to sample the original instance
        feasible_rows .&= .!identical_rows

        feasible_space[gidx] = combine(
                groupby(
                    data[feasible_rows, feat_names],
                    feat_names),
                nrow => :count
                )

        feasible_space[gidx].distance = distanceFeatureGroup(feasible_space[gidx], orig_instance, group)
    end

    return feasible_space
end
