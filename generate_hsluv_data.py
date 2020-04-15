import hsluv # from https://github.com/hsluv/hsluv-python
import numpy as np

npoints = 255

hsluv_data = []
hpluv_data = []

hues = np.linspace(0, 360, npoints, endpoint=False)

for h in hues:

    hsluv_data.append(hsluv.hsluv_to_rgb([h, 100., 60.]))
    hpluv_data.append(hsluv.hpluv_to_rgb([h, 100., 60.]))

hsluv_data = np.array(hsluv_data)
hpluv_data = np.array(hpluv_data)

np.savetxt('hsluv.txt', hsluv_data, fmt='%.15g')
np.savetxt('hpluv.txt', hpluv_data, fmt='%.15g')

    

