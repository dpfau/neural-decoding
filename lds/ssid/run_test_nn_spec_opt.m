load /Users/davidpfau/Dropbox/data/data.mat

[s,c,d] = bin(spikes,0.05,{mot1,mot2});

u = s(4001:6000,:)';
y = d(4001:6000,:)';

u = u([1:13,15:19,21:125],:);

[spec s0] = test_nn_spec_opt(y,u,15,[25 30]);