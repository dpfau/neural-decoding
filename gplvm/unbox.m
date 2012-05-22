% assign variable names to params in cell array

paramnames = { 'a', 'b', 'c', 'w', 'z' };

for i = 1:6
    eval([paramnames{i} ' = params{' int2str(i) '};']);
end