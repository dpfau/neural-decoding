y = randn(10,20);
d = 5;
params = rand(113,1);
sgplvm = ScaledGPLVM(y,d);
sgplvm.test_grad(params);