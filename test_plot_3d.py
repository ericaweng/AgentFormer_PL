import plotly.graph_objects as go
import numpy as np

# Create data
t = np.linspace(0, 10, 100)
x = np.sin(t)
y = np.cos(t)
z = t

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])

# Add frames for animation
frames = [go.Frame(data=[go.Scatter3d(x=x[:k], y=y[:k], z=z[:k])]) for k in range(1, len(t))]
fig.frames = frames

# Animation settings
fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(label="Play", method="animate", args=[None])])])

# Save to HTML
fig.write_html("../viz/animation.html")

# Save to MP4 (requires ffmpeg)
fig.write_html("../viz/animation.html")
fig.write_html("../viz/animation.html", auto_open=True)
fig.write_image("../viz/animation.png")
