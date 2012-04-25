function [s,c,d] = clean(s,c,d)
first = find(c(:,1)~=0&~isnan(c(:,1))&~isinf(c(:,1)),1);
last  = find(c(:,1)~=0&~isnan(c(:,1))&~isinf(c(:,1)),1,'last');
s = s(first:last,:);
c = c(first:last,:);
d = d(first:last,:);