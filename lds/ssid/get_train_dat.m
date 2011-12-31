load /Users/davidpfau/Dropbox/data/data.mat
addpath '/Users/davidpfau/Documents/Paninski Group/git-repo/util'

[s,c,d] = bin(spikes,0.05,{mot1,mot2});

y = d(4001:8000,:)';
u = s(4001:8000,:)';
u = u([1:13,15:19,21:125],:); % eliminate zeros