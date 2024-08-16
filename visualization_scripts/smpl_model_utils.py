import plotly.graph_objects as go
import plotly.subplots as subplots
from PIL import Image
import numpy as np
import tempfile


def up_layout(verts):
    verts_no_nan = np.nan_to_num(verts)

    center = (verts_no_nan.min(axis=(0, 1)) + verts_no_nan.max(axis=(0, 1))) / 2
    width_xyz = (verts_no_nan.max(axis=(0, 1)) - verts_no_nan.min(axis=(0, 1)))
    width = width_xyz.max()
    dim_min_x = center[0] - width / 2
    dim_max_x = center[0] + width / 2
    dim_min_y = center[1] - width / 2
    dim_max_y = center[1] + width / 2
    dim_min_z = center[2] - width / 2
    dim_max_z = center[2] + width / 2

    layout = dict(
        xaxis=dict(nticks=4, range=[dim_min_x, dim_max_x]),
        yaxis=dict(nticks=4, range=[dim_min_y, dim_max_y]),
        zaxis=dict(nticks=4, range=[dim_min_z, dim_max_z]),
        aspectmode='cube'
    )
    return layout


def display_smpl_model_plotly(model_info, title=None, with_joints=False, only_joints=False, show=False, savepath=None, scene_limits_verts=None):
    verts, joints, faces, kintree_table = model_info['verts'], model_info['joints'], model_info['faces'], model_info['kintree_table']

    # Define three viewing angles
    views = [
        dict(eye=dict(x=2, y=2, z=1.5)),
        dict(eye=dict(x=0.5, y=2.5, z=1)),
        dict(eye=dict(x=2.5, y=-1, z=1))
    ]

    fig = subplots.make_subplots(rows=1, cols=3, specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])

    # Add subplot for each view
    for idx, view in enumerate(views, start=1):
        # Plot mesh for each human
        face_color = 'rgba(141, 184, 226, 1)'  # Adjust alpha for transparency
        if not only_joints:
            for human_verts, human_faces in zip(verts, faces):
                x, y, z = human_verts.T
                i, j, k = human_faces.T
                fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=face_color, showscale=False),
                              row=1, col=idx)

        # Plot joints if needed
        if with_joints or only_joints:
            draw_skeleton_plotly(joints, kintree_table, fig, row=1, col=idx)

    if not scene_limits_verts:
        scene_layout = up_layout(verts)
    else:
        scene_layout = up_layout(scene_limits_verts)

    # Update scenes for each subplot with the corresponding view
    for idx, view in enumerate(views, start=1):
        scene_key = f'scene{"" if idx == 1 else idx}'  # Scene keys are 'scene', 'scene2', 'scene3', ...
        fig.update_layout({scene_key: {'camera': view, 'aspectmode': 'cube', 'xaxis': scene_layout['xaxis'], 'yaxis': scene_layout['yaxis'], 'zaxis': scene_layout['zaxis']},
                           'title': go.layout.Title(
                                    text=title,
                                    xref="paper",
                                    font=dict(size=20),
                                    x=0),
                           'grid': dict(rows=1, columns=3, pattern='independent'),
                           })

    # Save or show
    if savepath:
        fig.write_image(savepath, width=3000, height=1000)  # Adjusted for 3 subplots
        im = Image.open(savepath)
    else:
        with tempfile.TemporaryDirectory() as output_dir:
            temp_savepath = f"{output_dir}/temp.png"
            fig.write_image(temp_savepath, width=3000, height=1000)  # Adjusted for 3 subplots
            im = Image.open(temp_savepath)
    if show:
        fig.show()

    return im


def draw_skeleton_plotly(joints, kintree_table, fig, row, col):
    colors = ['red']#, 'green', 'blue']
    for joints3D in joints:
        fig.add_trace(go.Scatter3d(
            x=joints3D[:, 0],
            y=joints3D[:, 1],
            z=joints3D[:, 2],
            mode='markers+text',
            marker=dict(size=2),
            text=[f"{i}" for i in range(joints3D.shape[0])],
                showlegend=False,
        ), row=row, col=col)
        # for joints3D, kintree_table in zip(joints, kintree_tables):
        for i in range(0, kintree_table.shape[0]):
            j1, j2 = kintree_table[i]
            fig.add_trace(go.Scatter3d(
                x=[joints3D[j1, 0], joints3D[j2, 0]],
                y=[joints3D[j1, 1], joints3D[j2, 1]],
                z=[joints3D[j1, 2], joints3D[j2, 2]],
                mode='lines',
                # mode='lines+markers+text',
                line=dict(color=colors[i%len(colors)], width=2),
                    showlegend=False,
                # marker=dict(size=5),
                #     text=[f"{j1}", j2]
            ), row=row, col=col)

# Example usage
# model_info = ... # Define your model_info dictionary here
# display_smpl_model_plotly(model_info, with_joints=True)
