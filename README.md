# Jasmine Ng's Programming Projects

Welcome! This repository contains my work in the past years. It currently only contains my **Postgraduate Projects**, **Undergraduate Projects** and **External Projects**.  Below is a brief description of each project and the corresponding links to the report/code. For more information on each project, please visit the README file for each directory, where installation and usage steps are outlined.

## Postgraduate Projects

### Thesis: British Antarctic Survey Python Package

**DELPHI** (Detection of Laser Points in Huge image collections using Iterative learning) is a Python package developed in collaboration with the British Antarctic Survey, based on the methodology introduced by Timm Schoening.

The DELPHI package enables researchers to train and apply a laser point detection pipeline to benthic imagery. By providing paths to annotated laser point images and corresponding training images, users can initiate the training process. Once trained, the model can then be used to detect laser points in unlabelled (unseen) imagery. 

**Package available** [here](https://github.com/JasmineNg-codes/My-projects/tree/main/Postgraduate_projects/British_Antarctic_Survey_Python_Package).

**Full Python Code available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Postgraduate_projects/British_Antarctic_Survey_Python_Package/DELPHI/LPdetection.py).

**Report available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Postgraduate_projects/British_Antarctic_Survey_Python_Package/report/report.pdf).

### Machine Learning: MNIST Classification and LLM for time series forecasting

#### 1. MNIST Summed Digits Classification 

MNIST digit classification is a well-known problem in machine learning. This project explores a more challenging variation: images are augmented by placing two digits side by side, and the task is to predict the sum of the two digits rather than their individual identities.

The project first builds, trains, and tests a fully connected neural network on this augmented dataset. t-SNE is then used to visualize how the network represents and separates digit pairs at each layer. Finally, the performance of the neural network is compared with classical machine learning models, including SVM, random forests, and single decision trees, to evaluate their effectiveness on this summation task.

**Full Python Code available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Postgraduate_projects/Machine_Learning/MNIST_Inference_Pipelines/m1_workbook.ipynb).

**Report available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Postgraduate_projects/Machine_Learning/MNIST_Inference_Pipelines/report/M1%20(6).pdf).

#### 2. LLM for Time Series Forecasting

Time series forecasting is typically approached with regression models or, for more complex scenarios, neural networks. This project investigates whether a large language model (LLM) can perform time series forecasting, using Qwen-2.5 with a low-rank adaptation (LoRA) to predict prey–predator interactions. Due to computational constraints, the number of FLOPs (floating-point operations) is tracked at each layer of the LLM during processing. The project also delves into the inner workings of the model, including query, key, and value projections, multihead attention, residual connections, and activation functions, while systematically tuning the learning rate, context length, and LoRA rank to evaluate forecasting performance.

**Full Python Code available** [here](https://github.com/JasmineNg-codes/My-projects/tree/main/Postgraduate_projects/Machine_Learning/LLM_Time_Series_Forecasting/src).

**Report available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Postgraduate_projects/Machine_Learning/LLM_Time_Series_Forecasting/report/main.pdf).

### Statistics: Signal processing project

Measurements often contain a mixture of signal and background, which can be challenging when the goal is to isolate the signal. This project addresses this problem by first assuming probability models for both the signal and the background. We then use an extended maximum likelihood estimator (EMLE) to estimate the parameters of the signal model, effectively fitting the model to the observed data. The observations themselves are generated using an accept–reject method. Finally, the bias and uncertainty of the parameter estimates are evaluated through bootstrapping experiments.

**Full Python Code available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Postgraduate_projects/Statistics/S1.ipynb).

**Report available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Postgraduate_projects/Statistics/report/S1%20(4).pdf).

### Image Analysis: Exploring image classification, reconstruction and quality assessment

This project explores three interconnected topics: (1) classical image classification, (2) machine learning for image reconstruction, and (3) image quality assessment, with a focus on the pitfalls of using machine learning. Various image processing and restoration techniques are investigated. Segmentation is performed using the watershed and GrabCut algorithms. For denoising and inpainting of blurred or corrupted images, a Plug-and-Play Alternating Direction Method of Multipliers (PnP-ADMM) framework is applied. The performance of these methods is evaluated using metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM), while critically examining the limitations of these metrics in capturing perceptual image quality.

**Full Python Code available** [here](https://github.com/JasmineNg-codes/My-projects/tree/main/Postgraduate_projects/Image_Analysis).

**Report available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Postgraduate_projects/Image_Analysis/report/report.pdf).

### Medical Imaging: exploring PET, CT, MRI

This coursework explores three topics from the Medical Imaging modules: PET/CT reconstruction, MRI denoising, and CT segmentation. CT reconstruction is performed using Filtered Backprojection, OS-SART, and SIRT techniques, with a visual comparison to identify the most effective method. MRI denoising is evaluated using Gaussian, bilateral, mean, and frequency-based filters to determine their suitability for noise reduction. CT segmentation is carried out using a threshold-based approach, illustrating its limitations. Finally, image feature extraction employs metrics such as energy, mean absolute deviation, and uniformity to quantify tumor intensity, which is then compared against malignant or benign labels for performance assessment.

**Full Python Code available** [here](https://github.com/JasmineNg-codes/My-projects/tree/main/Postgraduate_projects/Medical_Imaging).

**Report available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Postgraduate_projects/Medical_Imaging/report/jn492_report.pdf).

### High Performance Computing: Optimising and parallelising heat diffusion code

This project asks the question: Can I outsmart ChatGPT at heat diffusion? Starting from a simple 2D model built by ChatGPT in C++, the goal is to speed it up by 2x using compiler tricks and parallelization techniques. Various optimization techniques like flattening and cache blocking were implemented and tuned, while parallelization approaches including MPI and OpenMP were explored through 2D Cartesian decomposition and experimentation with different ranks (MPI) and thread numbers (OpenMP). Each folder contains the optimised or parallelised code that yielded the best performance after tuning.

## Undergraduate Projects

### Using CoDa-PCA and linear models to understand Barium compositions 

**Results available in webpage format** [here](https://jasmineng-codes.github.io/barium_compositional_pca/).
**Full R Code available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/HIF_germany_internship_Rcode.Rmd).

**Objective**: To explore how trace elements in vegetation may serve as indicators for mineral ores. 

**Research question**: As a summer intern at HIF, I tidied the sample data from the Canadian Geological Survey database and addressed three main questions-
  1) From a compositional PCA biplot of vegetation trace elements, what elements are behaving differently from the rest?
  2) Is this a result of local geology and potential mineral ores?
  3) If not, what are the other potential drivers for the distinct elemental signal?
    
**Method**: Compositional data analysis (center log-ratio transformation), Principal component analysis, forward step-wise regression model
**Language**: R-studio

### Analyzing driver for bottom water hypoxia in Loch Etive

**Full Python Code available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Year_3_Environmental_Geoscience_LochEtive_Project.ipynb).

**Research question**: In my third year, my individual project in Practical geochemistry and data analysis addressed -
  1) Are Mn/Al and FeHR/FeT reliable indicators for sediment redox?
  2) Is N/C (a proxy for terrestrial or marine sediment) correlated to bottom water hypoxia?

**Method**: Linear regression, box plots, t-test
**Language**: Python

### Modelling past climate with oxygen isotopes

**Full Python Code available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Year_3_climate_modelling.ipynb)

**Research question**: In my third year, my coding assignment addressed how oxygen isotopes infer temperature changes from the Paleocene to the Pliocene.

**Method**: Linear regression
**Language**: Python

### Using CoDa-PCA to analyze the spatiotemporal change in trace metal pollutant in Glasgow top soils

**Full R Code available** [here](https://github.com/JasmineNg-codes/My-projects/blob/main/Year_4_Dissertation.R)

**Research question**: I am currently working on my 4th-year dissertation, which addresses:
  1) From a compositional PCA biplot of soil trace elements, what elements are clustering in urban areas?
  2) How does agricultural, industrial, recreational, and residential land use impact element compositions?
  3) What is the temporal change of selected urban elements between 2001, to 2018?
  4) What are the drivers for the spatial distribution of selected urban elements?

**Method**: Compositional data analysis (center log-ratio transformation), Principal component analysis, bar charts, ANOVA
**Language**: R

