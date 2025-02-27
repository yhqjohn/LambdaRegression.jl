"""One marker in a fixed AST template and the ordered scalar dependencies at that call."""
struct FixedScalarCall
    marker::Symbol
    dependencies::Tuple{Vararg{AstNode}}
end

FixedScalarCall(marker::Symbol, dependencies::AbstractVector{<:AstNode}) =
    FixedScalarCall(marker, Tuple(deepcopy(dependency) for dependency in dependencies))

"""
    FixedScalarScaffold(template, calls)

An explicitly authored scalar program scaffold. Each call marker must occur
exactly once in the template. The constructor owns structural copies; general
mystery extraction and automatic call discovery are deliberately outside this
type's contract.
"""
struct FixedScalarScaffold
    template::AstTree
    calls::Tuple{Vararg{FixedScalarCall}}
end

function FixedScalarScaffold(template::AstTree, calls::AbstractVector{<:FixedScalarCall})
    markers = getfield.(calls, :marker)
    length(unique(markers)) == length(markers) ||
        throw(ArgumentError("fixed scalar call markers must be unique"))
    return FixedScalarScaffold(deepcopy(template), Tuple(deepcopy(call) for call in calls))
end

"""A scaffold materialisation and the local scalar constants owned by it."""
struct ScalarMaterialization
    tree::AstTree
    local_consts::Dict{Symbol,Any}
    local_consts_ty::Dict{Symbol,TyLike}
end

"""Expected failure while validating or instantiating a fixed scaffold."""
struct ScalarScaffoldErr
    reason::Symbol
    marker::Union{Nothing,Symbol}
end

function _collect_constant_names!(names::Set{Symbol}, node::AstNode)
    if node isa ConstNode
        push!(names, node.name)
    elseif node isa AppNode
        _collect_constant_names!(names, node.func)
        _collect_constant_names!(names, node.arg)
    elseif node isa AbsNode
        _collect_constant_names!(names, node.body)
    elseif node isa LetNode
        _collect_constant_names!(names, node.value)
        _collect_constant_names!(names, node.body)
    end
    return names
end

function _rebuild_scalar_ast(replace_const, node::AstNode)
    if node isa ConstNode
        replacement = replace_const(node.name)
        return replacement === nothing ? node : replacement
    elseif node isa VarNode || node isa IndexNode
        return node
    elseif node isa AppNode
        func = _rebuild_scalar_ast(replace_const, node.func)
        func isa Err && return func
        arg = _rebuild_scalar_ast(replace_const, node.arg)
        arg isa Err && return arg
        return AppNode(func, arg)
    elseif node isa AbsNode
        body = _rebuild_scalar_ast(replace_const, node.body)
        body isa Err && return body
        return AbsNode(node.name, body)
    elseif node isa LetNode
        value = _rebuild_scalar_ast(replace_const, node.value)
        value isa Err && return value
        body = _rebuild_scalar_ast(replace_const, node.body)
        body isa Err && return body
        return LetNode(node.name, value, body)
    end
    return Err(ScalarScaffoldErr(:unsupported_template_node, nothing))
end

function _instantiate_candidate(candidate_template::AstNode, dependencies,
        dependency_names::Dict{Symbol,Int})
    return _rebuild_scalar_ast(candidate_template) do name
        index = get(dependency_names, name, 0)
        index == 0 && return nothing
        index <= length(dependencies) ||
            return Err(ScalarScaffoldErr(:missing_dependency, name))
        return deepcopy(dependencies[index])
    end
end

"""
    materialize_scalar_scaffold(scaffold, candidate; constant_prefix=:SCALAR_C)
        -> ScalarMaterialization | Err

Build an ordinary AST without mutating the scaffold or scalar expression. A
candidate is converted once, so every repeated call shares the same named
local constants. Each call then receives its own hygienic dependency copies.
The returned dictionaries belong to the materialisation.
"""
function materialize_scalar_scaffold(scaffold::FixedScalarScaffold,
        candidate::ScalarExpr; constant_prefix::Symbol=:SCALAR_C)
    required_features = _required_features(candidate)
    for call in scaffold.calls
        length(call.dependencies) >= required_features ||
            return Err(ScalarScaffoldErr(:missing_dependency, call.marker))
    end

    reserved = _collect_constant_names!(Set{Symbol}(), scaffold.template.root)
    dependency_symbols = Symbol[]
    dependency_nodes = AstNode[]
    for index in 1:required_features
        name = Symbol("__scalar_dependency_", index)
        while name in reserved
            name = Symbol("_", name)
        end
        push!(reserved, name)
        push!(dependency_symbols, name)
        push!(dependency_nodes, ConstNode(name))
    end
    dependency_names = Dict(name => index for (index, name) in pairs(dependency_symbols))

    local_consts = Dict{Symbol,Any}()
    local_consts_ty = Dict{Symbol,TyLike}()
    constant_index = 0
    function materialize_constant(value)
        while true
            constant_index += 1
            name = Symbol(string(constant_prefix), "_", constant_index)
            name in reserved && continue
            push!(reserved, name)
            local_consts[name] = value
            local_consts_ty[name] = FloatTy
            return ConstNode(name)
        end
    end
    candidate_template = scalar_to_ast(candidate, dependency_nodes;
        materialize_constant)
    candidate_template isa Err && return candidate_template

    calls = Dict(call.marker => call for call in scaffold.calls)
    counts = Dict(marker => 0 for marker in keys(calls))
    resolving = Set{Symbol}()
    function resolve_marker(name)
        call = get(calls, name, nothing)
        call === nothing && return nothing
        counts[name] += 1
        name in resolving && return Err(ScalarScaffoldErr(:cyclic_dependency, name))
        push!(resolving, name)
        dependencies = AstNode[]
        for dependency in call.dependencies
            resolved = _rebuild_scalar_ast(resolve_marker, deepcopy(dependency))
            if resolved isa Err
                delete!(resolving, name)
                return resolved
            end
            push!(dependencies, resolved)
        end
        delete!(resolving, name)
        return _instantiate_candidate(candidate_template, dependencies, dependency_names)
    end
    rebuilt = _rebuild_scalar_ast(resolve_marker, scaffold.template.root)
    rebuilt isa Err && return rebuilt
    for marker in keys(calls)
        counts[marker] == 1 ||
            return Err(ScalarScaffoldErr(
                counts[marker] == 0 ? :missing_marker : :duplicate_marker,
                marker,
            ))
    end
    return ScalarMaterialization(AstTree(rebuilt), local_consts, local_consts_ty)
end
