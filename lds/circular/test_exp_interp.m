function test_exp_interp( map, prec, template )

ns = [1,10,100];
ex = zeros(30,length(map));
ex2 = zeros(30,length(map));

for n = 1:3
    for i = 1:10
        [ex((n-1)*10+i,:) ex2((n-1)*10+i,:)] = exp_interp(map,prec,template,ns(n));
    end
    subplot(3,2,2*n-1);
    plot(add_vector(ex((n-1)*10+(1:10),:)', -interp(map,template,0)')), title(['E[X], ' int2str(ns(n)) ' samples.']); 
    subplot(3,2,2*n)
    plot(add_vector(ex2((n-1)*10+(1:10),:)', -interp(map,template,0).^2')), title(['E[X^2], ' int2str(ns(n)) ' samples.']); 
    drawnow
end