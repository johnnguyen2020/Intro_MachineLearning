function [data, class_indx] = generateSamples(n_samples, prior, mu, sigma)

class_dist = rand(n_samples, 1);
thresh = cumsum([0; prior]);
n = numel(mu);

data = cell(n, 1);
class_indx = cell(n, 1);

for j = 1:n
    n_samplesClass = nnz(class_dist>=thresh(j) & class_dist<thresh(j+1));
    data{j} = mvnrnd(mu{j}, sigma{j}, n_samplesClass);
    class_indx{j} = ones(n_samplesClass,1) * j;
end

data = cell2mat(data);
class_indx = cell2mat(class_indx);

end