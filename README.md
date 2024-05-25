#Master thesis Jussi Boersma

The code used for making my masters thesis can be found in this repository. The data it self cannot be found here as it concerns sensitive patient data.

##Abstract
About 10 years ago a link was found between multiple patients with unexplained familial idiopathic
ventricular fibrillation (IVF). IVF is very dangerous as it can lead to sudden cardiac death. Subsequently research was done in which the common link between these patients was found to be an
excessive expression of the DPP6 gene on chromosome 7. Extensive genealogical tracing was done
to find 601 possible mutation carriers of which 286 were found haplotype positive through genetic
testing. A dataset was comprised of 12-lead ECGs of both DPP6 mutation positive (n=156) and negative patients (n=156). For each patient, the first recorded out-patient clinic ECG was used.
The main goal of this research was to develop both machine and deep learning (DL) models to predict
whether a patient is DPP6 gene positive or negative based on the ECG. Another goal was to create a
level of explainability based on potentially good performing models by using Grad-CAM to visualize
what ECG features are important for DPP6 classification. A multitude of models where developed
that include machine learning techniques, 1D CNNs, 2D CNNs and an LSTM. We also used three
separate data augmentation methods with the goal of enhancing possible features or increasing the
size of the dataset.
Our models had Area Under the Curve Receiver Operating Characteristics (AUC) ranging from 0.76-
0.88. The best performing models were the 1D CNN (AUC 0.87), 1D LSTM (AUC 0.88) and the
1D Transfer learning model (AUC 0.88). The best average models using the 1D signal outperformed
the best average models using the 2D signal. Using Grad-CAM for the 2D models we found that the
lateral part of the QRS complex in leads aVR, II, V5 and V6, among other activated ECG regions,
to be the most important for detecting DPP6. This matches with the inferred pathophysiological
mechanism that the DPP6 gene mutation affects.
