function [spike pos vel] = binAllJoints( pk, dt, kin )
kin = cellfun(@(x) @(x)[x(1,:);fill_inf(x(2,:))],kin,'UniformOutput',0);
[spike,pos,vel] = bin(pk,dt,cellfun(@transpose,kin,'UniformOutput',0));
% Clear zeros
idx = find(pos(:,1));
spike = spike(idx,:);
pos = pos(idx,:);
vel = vel(idx,:);