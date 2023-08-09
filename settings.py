import matplotlib.pyplot as plt

my_colors = [
    "#ee6c4d",  # burnt sienna
    "#508aa8",  # air-force blue
    "#c6dea6",  # tea green
    "#8e5572",  # Magenta haze
    "#fbacbe",  # Cherry blossom pink
    "#f0a202",  # Gamboge (yellow)
    "#011638",  # Oxford blue
    "#214e34",  # Cal Poly green
    "#d4c1ec",  # Thistle (light purple)
]

plt.rc("axes", prop_cycle=plt.cycler(color=my_colors))
