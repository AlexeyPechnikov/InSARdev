def set_dark_mode():
    """Set dark mode styling for matplotlib plots and Jupyter widgets.

    Example:
        from insardev import ui
        ui.set_dark_mode()
    """
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display
    
    # Set Matplotlib global defaults for a dark theme:
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'savefig.facecolor': 'black',
        'text.color': 'lightgray',
        'axes.labelcolor': 'lightgray',
        'xtick.color': 'lightgray',
        'ytick.color': 'lightgray',
        'axes.edgecolor': 'lightgray'
    })
    
    # inject custom CSS for ipywidgets and cell outputs
    display(HTML("""
        <style>
        /* Overall dark theme for containers, input widgets, and cell outputs */
        .widget-box,
        .widget-text, .widget-int-text, .widget-float-text,
        .widget-dropdown,
        .jp-InputPrompt,
        .cell-output-ipywidget-background,
        .cell-output-ipywidget-background * {
            background-color: #333333 !important;
            color: lightgray !important;
            border-color: #555555 !important;
            outline: none !important;
        }

        /* Widget labels remain white */
        .widget-label {
            color: white !important;
        }

        /* Buttons styled with a dark background and white text */
        .widget-button {
            background-color: #444444 !important;
            color: lightgray !important;
            border-color: #555555 !important;
        }

        /* Progress bar: enforce green background, text, and border even within a dark container */
        .progress-bar,
        .cell-output-ipywidget-background .progress-bar {
            background-color: green !important;
            color: green !important;
            border-color: green !important;
        }

        /* For inner spans within progress bars (if any) */
        .progress-bar span,
        .cell-output-ipywidget-background .progress-bar span {
            color: green !important;
        }    

        output-ipywidget-background * {
            color: #333333 !important;
        }
        </style>
    """))
