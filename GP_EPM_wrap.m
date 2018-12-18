function [] =GP_EPM_wrap(input_file_name, output_file_name)
    data = load(input_file_name);
    B = data.A;
    B = triu(B, 1);
    N = size(B, 2);
    K = double(data.K);
    IsDisplay=false;
    Datatype = data.Datatype;
    Modeltype = data.Modeltype;
    TrainRatio = double(data.TrainRatio);
    Burnin = double(data.Burnin);
    Collections = double(data.Collections);
    rng(0,'twister');
    [idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);
    [AUCroc,AUCpr,F1,Phi,r,ProbAve,mi_dot_k,output,z] = GP_EPM(B,K,idx_train,idx_test,Burnin,Collections, IsDisplay, Datatype, Modeltype);
    save(output_file_name, 'z', 'mi_dot_k');
end