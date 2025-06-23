# Chemical Shift Prediction

*updating based on free time

This repo includes work on chemical shift prediction.


### What is a chemical shift?
In nuclear magnetic resonance (NMR) spectroscopy, the chemical shift describes the variation in the resonance frequency of a nucleus relative to a standard reference compound. This variation arises because the effective magnetic field experienced by a nucleus is influenced by its local electronic environment. Electrons surrounding the nucleus shield it from the main external magnetic field, and the degree of this shielding depends on the density and type of surrounding chemical bonds and functional groups. Consequently, chemically non-equivalent nuclei within the same molecule resonate at slightly different frequencies. These differences are measured against a reference standard, typically tetramethylsilane (TMS), whose protons are assigned a chemical shift value of 0 parts per million (ppm). The resulting chemical shift value, expressed in ppm, is a fundamental parameter in NMR that provides critical information for determining the molecular structure of a chemical compound.

![image](https://github.com/user-attachments/assets/c6261b76-8d0e-4ba9-a2ab-0e8bc200de78)

![image](https://github.com/user-attachments/assets/9bd16e06-d171-4c81-a58e-692e983a794f) = 1[G]


### BMRB dataset

"BMRB collects, annotates, archives, and disseminates spectral and quantitative data derived from NMR spectroscopic investigations of biological macromolecules and metabolites."

https://bmrb.io


### Force Fields

A force field is a collection of mathematical equations and associated parameters that describe the potential energy of a system of atoms as a function of their positions. It is essentially a computational model that calculates the forces acting on each atom, which in turn governs their motion and interactions.

#### MACE
MACE is a a machine learning software for predicting many-body atomic interactions and generating force fields. https://mace-docs.readthedocs.io/en/latest/index.html

Just so it happens that mace descriptors of 3D protein structures are highly informative. https://mace-docs.readthedocs.io/en/latest/guide/descriptors.html

Plotting UMAPs of mace descriptors against several protein properties yields some isles. (see in figures).


