addpath /Users/davidpfau/Documents/Code/lds
load /Users/davidpfau/Dropbox/data/data.mat

vn = {'A','B','C','Q','R','x0','x','y'};
for i = 1:8
    eval([vn{i} ' = cell(20,1);'])
end

[s,c,d] = bin(spikes,0.05,{mot1,mot2});

dat = [s(4001:8000,:),d(4001:8000,:)];
dat = dat - ones(4000,1)*mean(dat);
dat = dat(:,[1:13,15:end]);
dat = dat(:,[1:18,20:end]);

for i = 1:20
    [A{i},B{i},C{i},Q{i},R{i},x0{i},~,~,~] = ldsi(dat(1:2000,:),123,i,2000,500,1e-4);
    [x{i},y{i}] = gen(A{i},B{i},C{i},zeros(2,123),x0{i},dat(:,1:123)',[],[],[]);
end

for i = 1:20
    disp(['MSE(' int2str(i) ') = ' num2str(norm(dat(2001:end,124:125)'-y{i}(:,2001:end)))])
end