# Biometrics
This is the biometric project of:
- Filippo Olimpieri, 1933529
- Noemi Giustini, 1933541
- Ludovica Garufi, 1962596
- Filippo Guerra, 1931976

This project is created to implement iris-recognition and aims to use a known dataset to create software that is able to distinguish between eye images.
In particular, we are using cv2 functions to remove eyelashes, eyelid and other unwanted items and Gabor filters to conver the remaining iris to a unique code.
Finally, each eye's hamming code is calculated and matched against other images to find which one gives a positive result and which a negative.
This is crucial for identification.
