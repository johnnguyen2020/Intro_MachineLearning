function [classification] = inferClassLabel(data,mu,sigma,prior)

classification = double(mvnpdf(data,mu{1},sigma{1}) * prior(1) - mvnpdf(data,mu{2},sigma{2})* prior(2) > 0);
classification(classification==0) = 2;

end