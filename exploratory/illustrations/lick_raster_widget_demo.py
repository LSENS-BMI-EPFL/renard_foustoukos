#%%
# Jupyter / VS Code Interactive demo using ipympl (%matplotlib widget)
# Save this file as `exploratory/illustrations/lick_raster_widget_demo.py`.
# Open it in Jupyter, JupyterLab, or VS Code Interactive Window and run the cells.

# IPython magic: select the interactive widget backend provided by ipympl
%matplotlib widget

import matplotlib.pyplot as plt
import numpy as np

# sample data
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

# create interactive figure
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("ipympl interactive demo")
ax.set_xlabel("x")
ax.set_ylabel("y = sin(x^2)")

# show will display the interactive widget in the notebook / interactive window
plt.show()

#%%
# Fallback notes (run in a normal Python session if widget backend not available):
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# plt.plot(x, y)
# plt.savefig('demo_plot.png')

#%%
# Troubleshooting:
# - Ensure `ipympl` is installed in the same environment as the kernel you're using.
# - In JupyterLab (>=3) ipympl is prebuilt; for older versions you may need the labextension.
# - In VS Code Interactive Window, run "Run Current File in Interactive Window" with the correct kernel.
