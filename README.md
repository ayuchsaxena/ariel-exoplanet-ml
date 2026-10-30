# ARIEL Exoplanet ML

A collection of our work for the **NeurIPS – Ariel Data Challenge 2024**, focused on predicting exoplanet transit depths from the challenge data.

## Result

**Competition Silver Medalist — 37th out of 1,151 teams**

Awarded by Kaggle on **November 1, 2024**.

![Kaggle Silver Medal — NeurIPS Ariel Data Challenge 2024](kaggle-silver-medal.png.png)

First of all, I would like to thank the hosts for the preparations, my teammates @imayushsaxena and @octaviograu and all the competitors who made it that challenging!

Our top solutions
The solution got

variant	public	private	LB place	Final place
10th	0.6152848	0.6198108	53	-
11th	0.6128403	0.6293136	-	37
Summary of 11th
We used only AIRS data for the solution

The basic flow of the solution:

Standard correction, but without binning / shrinking the flux
Normalize the signal
Denoise the signal with gaussian_filter1d
Detect transit phases, including limb darkening
Calculate features for the whole signal
Calculate features for binned wavelengths by grouping nearest 6, 10, and 25 wavelengths.
On top of the features 1000 models were trained
The mean of the model's predictions gives the transit depth prediction
Sigma is based on the std for the model's predictions per wavelength
Link to the 11th solution.

Backbone pipeline
We tried many denoising strategies, so we needed a simple way to iterate. We built a functional pipeline that's easy to configure:

from functools import reduce

full_pipeline = [
    lambda  **kwargs: bin_wavelengths(**kwargs),
    lambda  **kwargs: normalize_signal(**kwargs),
    lambda **kwargs: filter_signal_with_gaussian(sigma=25, **kwargs),
    lambda **kwargs: polynomial_detrend(**kwargs),
    lambda **kwargs: fit_transit_poly(**kwargs),
    lambda **kwargs: collect_features(**kwargs),
]

initial_kwargs = {'signal': None, 'margin': 1, 'from_wavelength':39, 'to_wavelength':321}
result = reduce(lambda kw, func: func(**kw), full_pipeline, initial_kwargs)
Then we can combine result['features'] for different pipelines.

Each step in the pipeline can be organized like this:

def filter_signal_with_median(window=5, **kwargs):
    signal = kwargs['signal']
    filtered_signal = median_filter(signal, size=window)
    kwargs['signal'] = filtered_signal
    return kwargs
Transit phase detection
After experimenting with the BATMAN library, we found that there are several possible structures of the transit flux signal, even when fully denoised and detrended. Limb darkening can influence the calculation of transit depth, affecting how we calculate the transit edges

 def get_transit_phases2(**kwargs):
    signal = kwargs['signal'] + 1

    grad1 = np.gradient(signal)
    mn = np.argmin(grad1)
    mx = np.argmax(grad1)

    peaks = find_peaks(grad1, width=25)[0]
    ingress_left = peaks[(peaks < mn) & (mn - peaks <= 250)][-1] if np.any((peaks < mn) & (mn - peaks <= 250)) else mn-110
    ingress_right = peaks[(peaks > mn) & (peaks - mn <= 250)][0] if np.any((peaks > mn) & (peaks - mn <= 250)) else mn+110

    dips = find_peaks(-grad1, width=25)[0]
    egress_left = dips[(dips < mx) & (mx - dips <= 250)][-1] if np.any((dips < mx) & (mx - dips <= 250)) else mx-110
    egress_right = dips[(dips > mx) & (dips - mx <= 250)][0] if np.any((dips > mx) & (dips - mx <= 250)) else mx+110

    kwargs['ingress_left'] = ingress_left
    kwargs['ingress_right'] = ingress_right
    kwargs['egress_left'] = egress_left
    kwargs['egress_right'] = egress_right

    return kwargs
Features calculations
Knowing the ingress and egress of the transit, we fit a polynomial into the flux signal (ignoring the transit) and then perform detrending.

Then we can assume that once the plain detrended signal is on the level of constant 1, then we can derive a transit depth by the formula:

def transit_formula(s):
    return 1/s - 1
Here s can be a mean flux in the transit, flux at the middle point, etc. We collect quite a few here. This was a huge speed improvement compared to the minimize method.

Additionally, we collect generic features on the in-transit flux, like the standard deviation.

Models training
We have generated 500 Ridge models and 500 PLSRegression models that formed the ensemble.

Alpha for Ridge was random:

alpha = 10 ** (2 - 3 * np.random.rand())
The same as n_components for PLSRegression:

model = PLSRegression(n_components=n_components, tol=1e-8)
We used different subsets of the training data for each model:

subset_size = int(X_train.shape[0] * np.random.uniform(0.5, 1))
We sorted the model predictions for each wavelength, removed the lowest and highest 30%, and took the mean of the remaining predictions.

Sigma calculation
We calculated the standard deviation of the sorted predictions for each wavelength:

std_preds = np.std(sorted_preds, axis=0)
We then multiplied the standard deviation by a coefficient that worked best in LB:

COEFF_0 = 4 # This one worked the best
sigma_response = sigma * COEFF_0
What worked well, though didn't make it to the final submission
Usage of wavelets (pywt) - our favorite one was db29
median_filter
savgol_filter
What didn't work
Gaussian Process Regression - we tried different kernels here, but didn't see breakthroughs while a big increase of the processing time
LinearRegresion, SVM, LGBM, CatBoost, RandomForest, XGBoost…
BayesianRidge - worked well, though long time (and didn't fit into the memory for more features)
We tried to train a neural network with a loss function for both sigma and transit depth (the competition score), but didn't have much luck due to the limited time
Improvements We Lacked Time For
Generate the train data with BATMAN
Use FGS data
Use data for real molecules to generate additional training data
Collect more meaningful features (and filter out those which don't help)
Sigma approaches - we didn't use all the potential here
Please feel free to explore our best solution.

Authors
Shamil Yagiyayev
shamily
Octavi Grau
octaviograu
Ayush Saxena
ayuchsaxena 

Share
Citation
Shamil Yagiyayev, Octavi Grau, Ayush Saxena. 37th place solution. https://www.kaggle.com/competitions/ariel-data-challenge-2024/writeups/wavelength-masters-37th-place-solution. 2024. Kaggle


Our team placed **37th among 1,151 teams**, earning a **Competition Silver Medal** in the NeurIPS – Ariel Data Challenge 2024.

