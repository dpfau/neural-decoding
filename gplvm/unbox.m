% assign variable names to params in cell array

for var = fieldnames(params)'
    eval([char(var) ' = params.' char(var)])
end