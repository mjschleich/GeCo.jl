
struct PLAFProgram
    groups::Vector{Tuple{Vararg{Symbol}}}
    constraints::Vector{Pair{Array{Symbol, 1}, Function}}
end

initPLAF() = PLAFProgram(Vector{Tuple}(), Vector{Expr}())

Base.empty!(p::PLAFProgram) = begin
    empty!(p.groups)
    empty!(p.constraints)
    p
end

plaf_helper(x, t) = begin
    constraints = quote $(ground.(t)...) end
    ret = quote
        $push!($x.constraints, $constraints)
    end
    println(ret)
    ret
end

group_helper(x, t) = begin
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
replace_syms!(x, membernames) = x
replace_syms!(q::QuoteNode, membernames) = replace_syms!(Meta.quot(q.value), membernames)

function replace_syms!(e::Expr, membernames)
    # if onearg(e, :^)
    #     e.args[2]
    # elseif onearg(e, :cols)
    #     addkey!(membernames, :($(e.args[2])))
    if e.head == :quote
        addkey!(membernames, Meta.quot(e.args[1]) )
    elseif e.head == :.
        if e.args[1] ∈ (QuoteNode(:cf), QuoteNode(:x_cf), QuoteNode(:counterfactual))
            return replace_syms!(e.args[2], membernames)
        elseif e.args[1] ∈ (QuoteNode(:x), QuoteNode(:inst), QuoteNode(:instance))
            # TODO: Check if this is correct
            return quote __orig_instance.$(e.args[2].value) end
        else
            @error "The tuple identifier should be one of (cf, x_cf, counterfactual) or (x, inst, instance), we got $(e.args[1])"
        end
    else
        e2 = mapexpr(x -> replace_syms!(x, membernames), e)
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

function ground(kw)
    grounded_constraints = Pair{Vector{Symbol}, Any}[]

    membernames = Dict{Any, Symbol}()

    body::Expr = replace_syms!(kw, membernames)
    source::Expr = Expr(:vect, keys(membernames)...)
    inputargs::Expr = Expr(:tuple, values(membernames)...)

    generated_func = quote
        GeCo.make_source_concrete($(source)) => __orig_instance -> $inputargs -> $body
    end
    
    generated_func
end

