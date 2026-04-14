## Fair, Good, Very Good, Premium, Ideal
## I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)
## D (best) to J (worst)

import pandas as pd


df = pd.read_csv("data/diamonds.csv")
mapa_calidad = {"Fair" : 1, "Good" : 2, "Very Good" : 3, "Premium" : 4, "Ideal" : 5}
mapa_claridad = {"I1" : 1, "SI2" : 2, "SI1" : 3, "VS2" : 4, "VS1" : 5, "VVS2" : 6, "VVS1" : 7, "IF" : 8}
mapa_color = {"J" : 1, "I" : 2, "H" : 3, "G" : 4, "F" : 5, "E" : 6, "D" : 7}
