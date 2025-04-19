class UI:

    def __init__(self, mode: str = 'dark', dpi=150, delay: int = 1000):
        """
        Set dark mode styling for matplotlib plots and Jupyter widgets.

        Example:
            from insardev.UI import UI
            UI('dark')
        """
        import matplotlib.pyplot as plt
        from IPython.display import HTML, Javascript, display
        
        if mode not in ['dark', 'light']:
            raise ValueError("Invalid mode. Must be 'dark' or 'light'.")

        plt.rcParams['figure.figsize'] = [12, 4]
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['figure.titlesize'] = 24
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

        if mode != 'dark':
            return

        # set Matplotlib global defaults for a dark theme:
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
        
        # custom CSS for ipywidgets
        dark_css = """
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
            .cell-output-ipywidget-background .progress-bar {
                background-color: #cca700 !important;
            }
            /* progress-bar progress-bar-success */
            .cell-output-ipywidget-background .progress-bar-success {
                background-color: #4caf50 !important;
            }

            /* progress-bar progress-bar-danger */
            .cell-output-ipywidget-background .progress-bar-danger {
                background-color: #f44336 !important;
            }
                            
            /* For inner spans within progress bars (if any) */
            /*.progress-bar span,
            .cell-output-ipywidget-background .progress-bar span {
                color: green !important;
            }*/    

            output-ipywidget-background * {
                color: #333333 !important;
            }
            .cell-output-ipywidget-background {
                background: #333333 !important;
            }
            
            .jupyter-widgets .widget-html-content,
            .jupyter-widget-html-content {
                font-family: monospace !important;
            }
            </style>
        """
        
        # inject CSS
        display(HTML(dark_css))
        # inject CSS after delay
        if delay is not None:
            js = f"""
            setTimeout(function(){{
                document.head.insertAdjacentHTML('beforeend', `{dark_css}`);
            }}, {delay});
        """
        display(Javascript(js))
