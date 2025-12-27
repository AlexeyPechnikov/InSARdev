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
        plt.rcParams['figure.titlesize'] = 16
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

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
            .cell-output-ipywidget-background *:not([class*="leaflet"]):not([class*="pv-"]):not(canvas):not(svg):not(svg *) {
                background-color: #333333 !important;
                color: lightgray !important;
                border-color: #555555 !important;
                outline: none !important;
            }

            /* Exclude leaflet and pyvista elements from dark theme override */
            .leaflet-container,
            .leaflet-container *,
            [class*="leaflet-"],
            [class*="pv-"] {
                background-color: unset !important;
                color: unset !important;
                border-color: unset !important;
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
        
        # inject CSS with MutationObserver for dynamic widgets
        js = f"""
        (function(){{
            const css = `{dark_css}`;
            
            // Create and inject style element directly into head
            const styleElement = document.createElement('style');
            styleElement.textContent = css.replace(/<\\/?style>/g, '');
            document.head.appendChild(styleElement);
            
            // Apply styles immediately to existing elements
            document.querySelectorAll('.progress-bar-success').forEach(el => {{
                el.style.backgroundColor = '#4caf50';
            }});
            
            // Observer for dynamically created widgets
            const observer = new MutationObserver(function(mutations) {{
                mutations.forEach(function(mutation) {{
                    if (mutation.addedNodes.length) {{
                        mutation.addedNodes.forEach(function(node) {{
                            if (node.nodeType === 1) {{ // Element node
                                // Re-apply progress bar styling to new elements
                                node.querySelectorAll('.progress-bar-success').forEach(el => {{
                                    el.style.backgroundColor = '#4caf50';
                                }});
                                if (node.classList && node.classList.contains('progress-bar-success')) {{
                                    node.style.backgroundColor = '#4caf50';
                                }}
                            }}
                        }});
                    }}
                }});
            }});
            
            // Observe the entire document for changes
            observer.observe(document.body, {{
                childList: true,
                subtree: true
            }});
        }})();
        """
        display(Javascript(js))
        
        # Monkey-patch xarray to inject dark theme CSS directly into its HTML repr
        # This ensures the CSS travels with the HTML into VS Code's isolated output frames
        try:
            import xarray as xr
            
            xarray_dark_css = """
            <style>
            .xr-wrap, .xr-wrap * { background-color: #2b2b2b !important; color: #d0d0d0 !important; border-color: #444 !important; box-shadow: none !important; }
            .xr-header { background-color: #1f1f1f !important; color: #e0e0e0 !important; }
            .xr-section-summary, .xr-section-inline-details { background-color: #2b2b2b !important; color: #d0d0d0 !important; }
            .xr-var-list, .xr-var-item { background-color: #2b2b2b !important; color: #d0d0d0 !important; }
            .xr-var-name, .xr-var-dims, .xr-var-dtype, .xr-var-preview { color: #d0d0d0 !important; }
            .xr-index-name { color: #d0d0d0 !important; }
            .xr-array-wrap { background-color: #252525 !important; }
            pre.xr-text-repr-fallback { background-color: #252525 !important; color: #cfcfcf !important; }
            </style>
            """
            
            # Patch DataArray._repr_html_ (only once - check for marker)
            if hasattr(xr.DataArray, '_repr_html_') and not getattr(xr.DataArray._repr_html_, '_dark_patched', False):
                _original_da_repr = xr.DataArray._repr_html_
                def _dark_da_repr(self):
                    html = _original_da_repr(self)
                    if html:
                        return xarray_dark_css + html
                    return html
                _dark_da_repr._dark_patched = True
                xr.DataArray._repr_html_ = _dark_da_repr
            
            # Patch Dataset._repr_html_ (only once - check for marker)
            if hasattr(xr.Dataset, '_repr_html_') and not getattr(xr.Dataset._repr_html_, '_dark_patched', False):
                _original_ds_repr = xr.Dataset._repr_html_
                def _dark_ds_repr(self):
                    html = _original_ds_repr(self)
                    if html:
                        return xarray_dark_css + html
                    return html
                _dark_ds_repr._dark_patched = True
                xr.Dataset._repr_html_ = _dark_ds_repr
        except ImportError:
            # xarray not installed
            pass
