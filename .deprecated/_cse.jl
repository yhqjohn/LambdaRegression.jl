# Common Subexpression Elimination for DeBruijn Trees

module CSEModule

using ..DeBruijnModule: DBNode, VarDB, AbsDB, AppDB, ProxyDB, traverse, Context, Abs, Right, NewLeft
import ..DeBruijnModule: pretty_string
using MLStyle
using TestItems

public VarNode, traverse_with_id
export cse_candidates


struct VarNode{T} <: DBNode{T}
    id::UInt
end
pretty_string(node::VarNode{T}) where {T} = "\$$(node.id)"


"""
    traverse_with_id(f, node::DBNode{T}, start_id=UInt(0)) where {T}

Traverse the node with a unique id for each abstraction.
Similar to `traverse`, but the first argument is a function that takes three arguments: the node, the depth, the next id, and the abstraction stack.
The abstraction stack is a stack of unique ids for each abstraction from the root to the current node.
"""
function traverse_with_id(f, node::DBNode{T}, start_id=UInt(0)) where {T}
    unique_id::UInt = start_id
    abs_stack = UInt[]
    stack = Context[]
    depth::UInt = UInt(0)
    result = nothing
    node = node
    while !isnothing(node) || !isempty(stack)
        if !isnothing(node)
            result = f(node, depth, unique_id, abs_stack)
            if isnothing(result)
                node = @match node begin
                    AbsDB(body) =>begin
                        push!(abs_stack, unique_id)
                        unique_id += 1
                        push!(stack, Abs)
                        depth += 1
                        body
                    end
                    AppDB(func, arg) =>(push!(stack, Right(arg)); func)
                    _ => (result= node; node = nothing)
                end
            else
                node = nothing
            end
        else
            result = @match pop!(stack) begin
                Abs => (depth -= 1; pop!(abs_stack); AbsDB{T}(result))
                Right(value) => begin
                    push!(stack, NewLeft(result))
                    node = value
                    result
                end
                NewLeft(value) => AppDB{T}(value, result)
            end
        end
    end
    return result
end

@testitem "traverse_with_id" begin
    using LambdaRegression.DeBruijnModule
    using LambdaRegression.ParseModule
    using LambdaRegression.DeBruijnModule.CSEModule: traverse_with_id

    # basic test, test it behave the same as traverse if f ignores the abs_stack
    foo = DBNode(@lambda (*)->(*)->(&3 * &1 * ((*)->&0 * &2)))
    new_foo = traverse_with_id(foo) do x, d, n, s
        if x isa VarDB
            VarDB(x.index + 1)
        else
            nothing
        end
    end
    @test new_foo == DBNode(@lambda (*)->(*)->(&4 * &2 * ((*)->&1 * &3)))
    new_foo2 = traverse_with_id(new_foo) do x, d, n, s
        if x isa VarDB
            ProxyDB(Int(x.index))
        else
            nothing
        end
    end
    @test new_foo2 == DBNode(@lambda (*)->(*)->(4 * 2 * ((*)->1 * 3)))

    ## λ.λ.((λ.λ.(#1 #3))(λ.λ.λ.(#1 #4)))
    l1 = DBNode(@lambda((*)->(*)->(((*)->(*)->(&0 * &2))*((*)->(*)->(*)->(&0 * &3)))))
    ds = []
    ns = []
    ss = []
    new_l1 = traverse_with_id(l1) do x, d, n, s
        push!(ds, d)
        push!(ns, n)
        push!(ss, copy(s))
        nothing
    end
    @test new_l1 == l1
    @test ds == UInt[0, 1, 2, 2, 3, 4, 4, 4, 2, 3, 4, 5, 5, 5]
    @test ns == UInt[0, 1, 2, 2, 3, 4, 4, 4, 4, 5, 6, 7, 7, 7]
    @test ss == [
        UInt[], # λ.λ.((λ.λ.(#1 #3))(λ.λ.λ.(#1 #4))) root 
        UInt[0], # λ.((λ.λ.(#1 #3))(λ.λ.λ.(#1 #4)))
        UInt[0, 1], # ((λ.λ.(#1 #3))(λ.λ.λ.(#1 #4)))
        UInt[0, 1], # λ.λ.(#1 #3)
        UInt[0, 1, 2], # λ.(#1 #3)
        UInt[0, 1, 2, 3], # #1 #3
        UInt[0, 1, 2, 3], # #1
        UInt[0, 1, 2, 3], # #3
        UInt[0, 1], # λ.λ.λ.(#1 #4)
        UInt[0, 1, 4], # λ.λ.(#1 #4)
        UInt[0, 1, 4, 5], # λ.(#1 #4)
        UInt[0, 1, 4, 5, 6], # #1 #4
        UInt[0, 1, 4, 5, 6], # #1
        UInt[0, 1, 4, 5, 6], # #4
    ]
end

"""
    conanicalize(node::DBNode{T}, abs_stack::Vector{UInt}, start_id=UInt(0)) where {T}

Conanicalize the sub-tree by replacing all free variables with their corresponding VarNode,
a node with index that referes to the unique id of the corresponding abstraction,
and all bound variables are kept as they are.
The position of the subtree in the tree is determined by the abs_stack and start_id.
"""
function conanicalize(node::DBNode{T}, abs_stack::Vector{UInt}, start_id=UInt(0)) where {T}
    return traverse_with_id(node, start_id) do x, d, n, s
        @match x begin
            VarDB(index) && if index >= d end => VarNode{T}([abs_stack; s][end-index])
            _ => nothing
        end
    end
end

"""
    cse_candidates(node::DBNode{T}) where {T}

Find all sub-trees except variables with the same canonical form,
and return a frequency dictionary that maps the canonical forms that appear more than once to their frequency.
"""
function cse_candidates(node::DBNode{T}) where {T}
    frequencies = Dict{DBNode{T}, UInt}()
    traverse_with_id(node) do x, d, n, s
        if !(x isa VarDB)
            conanical = conanicalize(x, s, n)
            if haskey(frequencies, conanical)
                frequencies[conanical] += 1
            else
                frequencies[conanical] = 1
            end
        end
        nothing
    end
    filter(((k, v),)->v >= 2, frequencies)
end

@testitem "cse_candidates" begin
    using LambdaRegression.ParseModule
    using LambdaRegression.DeBruijnModule
    using LambdaRegression.DeBruijnModule.CSEModule: cse_candidates, VarNode

    ## λ.λ.((λ.λ.(#1 #3))(λ.λ.λ.(#1 #4)))
    l1 = DBNode(@lambda((*)->(*)->(((*)->(*)->(&1 * &2))*((*)->(*)->(*)->(&1 * &3)))))
    @test cse_candidates(l1) == Dict([AbsDB(AbsDB(AppDB(VarDB(1), VarNode{Any}(1))))=>2])
end

function lca_by_canonical(node::DBNode{T}, canonical::DBNode{T}, freq::UInt) where {T}
    count::UInt = 0

end



end
