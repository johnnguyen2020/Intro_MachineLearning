function plotDecisionBoundary(xMin,xMax,yMin,yMax,mu,sigma,prior)

[X,Y] = meshgrid(linspace(xMin,xMax,100),linspace(xMin,xMax,100));
data_MESH = [X(:) Y(:)];
decision_boundary = mvnpdf(data_MESH,mu{1},sigma{1}) * prior(1) - mvnpdf(data_MESH,mu{2},sigma{2})* prior(2);
max_d = max(decision_boundary);
min_d = min(decision_boundary);

hold on
contour(X,Y,reshape(decision_boundary,100,100),[min_d*[0.05],0,[0.05]*max_d])
end