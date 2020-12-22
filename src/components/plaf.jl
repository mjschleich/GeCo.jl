
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

plaf_helper(x, t) = begin
    quote
        $append!($x.constraints, $t)
    end
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
