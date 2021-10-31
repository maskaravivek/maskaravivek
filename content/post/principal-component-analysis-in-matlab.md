---
title: "Principal Component Analysis in MATLAB"
author: "Vivek Maskara"
date: 2020-05-25T07:09:59.590Z
lastmod: 2021-10-29T21:25:37-07:00

description: ""

subtitle: ""

categories: [Machine Learning]

tags:
 - Machine Learning
 - PCA
 - MATLAB

image:
  caption: ""
  focal_point: "smart"
  preview_only: true

images:
 - "/post/img/2020-05-25_principal-component-analysis-in-matlab_0.jpg"
 - "/post/img/2020-05-25_principal-component-analysis-in-matlab_1.jpeg"
 - "/post/img/2020-05-25_principal-component-analysis-in-matlab_2.jpeg"
 - "/post/img/2020-05-25_principal-component-analysis-in-matlab_3.jpeg"
 - "/post/img/2020-05-25_principal-component-analysis-in-matlab_4.jpeg"
 - "/post/img/2020-05-25_principal-component-analysis-in-matlab_5.jpeg"
 - "/post/img/2020-05-25_principal-component-analysis-in-matlab_6.jpg"
 - "/post/img/2020-05-25_principal-component-analysis-in-matlab_7.jpg"
 - "/post/img/2020-05-25_principal-component-analysis-in-matlab_8.png"


aliases:
- "/principal-component-analysis-in-matlab-f5c27b76e8c"

---

Principal Component Analysis(PCA) is often used as a data mining technique to reduce the dimensionality of the data. In this post, I will show how you can perform PCA and plot its graphs using MATLAB.

#### What is PCA?

Principal Component Analysis(PCA) is a statistical method to reduce the dimensionality of the data. It assumes that data with large variation is important. PCA tries to find a unit vector(first principal component) that minimizes the average squared distance from the points to the line. Other components are lines perpendicular to this line.

#### Why do we need PCA?

Working with a large number of features is computationally expensive and the data generally has a small intrinsic dimension. To reduce the dimension of the data we will apply Principal Component Analysis(PCA) which ensures that no information is lost and checks if the data has a high standard deviation. Thus, PCA helps in fighting the curse of dimensionality and reduces the dimensionality to select just the top few features that satisfactorily represent the variation in data.

#### How PCA is done?

The method for PCA is as follows:

- Normalize the values of the feature matrix using normalize function in MATLAB
- Calculate the empirical mean along each column and use this mean to calculate the deviations from mean
- Next, we use these deviations to calculate the `p x p`covariance matrix.
- Next, find the eigenvectors and eigenvalues of the covariance matrix
- Sort the columns of the matrix in decreasing order of eigenvalues and compute the cumulative energy content for each eigenvector.
- Finally select a subset of the eigenvectors as the basis vectors and project the z-score of the data on the basis vectors.

### PCA in MATLAB

MATLAB provides a convenient way to perform PCA using the `pca` function. Read up more about it [here](https://www.mathworks.com/help/stats/pca.html). The method takes a `featureMatrix` as input and performs the PCA analysis on it.

```
featureMatrix = normalize(featureMatrix); % Normalize the feature matrix

% No of dimensions to keep
numberOfDimensions = 5;

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(featureMatrix); % Perform PCA analysis

reducedDimension = COEFF(:,1:numberOfDimensions);
reducedFeatureMatrix = featureMatrix * reducedDimension;
```

- `COEFF` is a p x p coefficient matrix. Each column of `COEFF` contains coefficients for one principal component and the columns are in descending order of component variance.
- `SCORE`: Principal component scores are the representations of `featureMatrix` in the principal component space.
- `LATENT`: Principal component variances, that is the eigenvalues of the covariance matrix of `featureMatrix`, returned as a column vector.
- `TSQUARED`: Hotelling’s T-squared statistic for each observation in `featureMatrix`.
- `EXPLAINED`: The percentage of the total variance explained by each principal component and mu, the estimated mean of each variable in `featureMatrix` .

Note: It is recommended to normalize the feature matrix before performing PCA. Normalizing the features avoid the results from getting skewed in favor of a feature that has bigger values.

#### Plotting the graphs

Now, that we have obtained the `reducedFeatureMatrix` and the `COEFF, SCORE, LATENT, TSQUARED, EXPLAINED` we will go ahead and plot these values.

**Eigen Vectors**

Use the following code snippet to plot the eigenvectors obtained after performing the PCA analysis.

```
figure('Name','EigenVectors','NumberTitle','off');
bar(COEFF);
saveas(gcf, 'EigenVectors', 'jpeg');
```

For our dataset, this is the plot that I got after plotting the eigenvectors.

![](/post/img/2020-05-25_principal-component-analysis-in-matlab_0.jpg#layoutTextWidth)

**Principal Components**

Use the following code snippet to plot the top 5 principal components obtained after performing the PCA analysis. The principal components obtained after PCA are in the order of their variances.

```
plotpca={'f1','f2','f3','f4','f5'};
for p=1:5
    figure('Name',plotpca{p},'NumberTitle','off');

    score_length = size(SCORE, 1);
    tx_vect= SCORE(1:score_length, p);
    scatter(1:score_length,tx_vect,'r')
    xlabel('Timestamp in order') 
    ylabel("Feature Values("+ plotpca{p}+")")
    saveas(gcf,strcat(feature_plots, strcat('Plot_',plotpca{p})), 'jpeg');
end
```

![](/post/img/2020-05-25_principal-component-analysis-in-matlab_1.jpeg#layoutTextWidth)![](/post/img/2020-05-25_principal-component-analysis-in-matlab_2.jpeg#layoutTextWidth)![](/post/img/2020-05-25_principal-component-analysis-in-matlab_3.jpeg#layoutTextWidth)![](/post/img/2020-05-25_principal-component-analysis-in-matlab_4.jpeg#layoutTextWidth)![](/post/img/2020-05-25_principal-component-analysis-in-matlab_5.jpeg#layoutTextWidth)

Note: In the plots for features `f1` to `f5` you can notice that the different features have different degrees of variance.

**Top Features**

The scatter plot of the first 3 principal components can be obtained by the following code snippet.

```
scatter3(SCORE(:,1),SCORE(:,2),SCORE(:,3))
saveas(gcf,strcat(feature_plots, strcat('Scatter_Plot_Top_3_Features')), 'jpeg');
hold off;
```

The plot below shows the first 3 principal components.

![](/post/img/2020-05-25_principal-component-analysis-in-matlab_6.jpg#layoutTextWidth)

**Variance**

The variance distribution plot is useful in deciding the number of principal components to keep. If the top 5 features show around 95% variance, we can ignore the other features as their contribution is very less.

The snippet below plots the variance distribution.

```
figure();
pareto(LATENT);
title('Variance Distribution')
saveas(gcf,strcat(feature_plots, strcat('Plot_','Variance Distribution')), 'jpeg');
```

**Variance Percentage**

The snippet below plots the variance percentage.

```
figure();
pareto(EXPLAINED);
title('Percentage of Variance Explained');
saveas(gcf,strcat(feature_plots, strcat('Plot_','Percentage of Variance Explained')), 'jpeg');
```

The plot shows the percentage of variance `Explained`.

![](/post/img/2020-05-25_principal-component-analysis-in-matlab_7.jpg#layoutTextWidth)

Note, that the distribution is not the ideal distribution as it is based on a real dataset.

You can buy me a coffee if this post really helped you learn something or fix a nagging issue!

* * *
Written on May 25, 2020 by Vivek Maskara.

Originally published on [Medium](https://medium.com/@maskaravivek/principal-component-analysis-in-matlab-f5c27b76e8c)
