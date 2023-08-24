""" 
    subscript(i::Integer) 

Set up string with integer `i` as subscript (for printing). 

From https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia 

# Examples
```julia-repl
julia> println("Studio"*subscript(45))
Studio₄₅
```
""" 
subscript(i::Integer) = i<0 ? error("$i is negative") : join('₀'+d for d in reverse(digits(i)))


""" 
    pad_digits(x) 

Returns a vector with string representations of the numbers in x such that they have the same number of digits, including trailing zeros.

Useful when printing numbers in latex tables and annotating figures.

# Examples
```julia-repl
julia> pad_digits([1.21, 13.3, 123.456])
3-element Vector{String}:
 "1.210"
 "13.300"
 "123.456"
```
""" 
function pad_digits(x)
    strs = string.(x)
    digits = [length(str[(findfirst.('.', str) + 1):end]) for str in strs]
    maxDigits = maximum(digits)
    lengthNumber = [length(str) for str in strs]
    for i = 1:length(strs)
        strs[i] = strs[i]*repeat("0",maxDigits-digits[i])
    end
    return strs
end

""" 
    find_min_matrix(matrix, k) 

Returns the Cartesian indices of the k:th smallest values in `matrix`. 

# Examples
```julia-repl
julia> A = [10 8 12; 5 4 9; 3 6 2]
julia> find_min_matrix(A, 3)
```
""" 
function find_min_matrix(matrix, k)
    flattened_matrix = vec(matrix)
    sorted_indices = sortperm(flattened_matrix)
    return CartesianIndices(size(matrix))[sorted_indices[1:k]]
end


""" 
    find_max_matrix(matrix, k) 

Returns the Cartesian indices of the k:th largest values in `matrix`. 

# Examples
```julia-repl
julia> A = [10 8 12; 5 4 9; 3 6 2]
julia> find_max_matrix(A, 3)
```
""" 
function find_max_matrix(matrix, k)
    flattened_matrix = vec(matrix)
    sorted_indices = sortperm(flattened_matrix, rev=true)
    return CartesianIndices(size(matrix))[sorted_indices[1:k]]
end