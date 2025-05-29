import pandas as pd
import plotly.express as px


def plot_data(df,
              x, y,
              animation_frame,
              hover_name,
              range_x,
              range_y,
              ):
    fig = px.scatter(df, x=x, y=y, animation_frame=animation_frame, animation_group=None,
                     size="charge",
                     # color=None,
                     hover_name=hover_name,
                     size_max=10,
                     range_x=range_x, range_y=range_y
                     )

    fig["layout"].pop("updatemenus")
    fig.update_layout(
        width=500,
        height=500,
        title="Gráfico Cuadrado Interactivo",
        margin=dict(l=50, r=50, t=50, b=50),
        autosize=True
    )
    fig.show()


if __name__ == "__main__":
    aa = pd.DataFrame([
        {"x": 1, "y": 1, "step": 10, "charge": 0.1},
        {"x": 3, "y": 3, "step": 10, "charge": 0.1},
        {"x": 2, "y": 2, "step": 11, "charge": 1},
        {"x": 3, "y": 3, "step": 12, "charge": 0.5},
        {"x": 4, "y": 4, "step": 13, "charge": 3},
    ])

    print(aa)
    plot_data(aa, x="x", y="y", animation_frame="step", hover_name="step",
              range_x=[8, 20], range_y=[8, 20])
