# This is needed to use requirejs on google colab
import IPython


def configure_plotly_browser_state_head():
    display(IPython.core.display.HTML('''
          <script>
            require.config({
              paths: {
                d3: '//cdnjs.cloudflare.com/ajax/libs/d3/3.4.8/d3.min',
                jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
              }
            });
          </script>
          '''
    ))


def configure_plotly_browser_state_model():
    display(IPython.core.display.HTML('''
          <script>
            require.config({
              paths: {
                d3: '//cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min',
                jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
              }
            });
          </script>
          '''
    ))


def configure_plotly_browser_state_neuron():
    display(IPython.core.display.HTML('''
          <script>
            require.config({
              paths: {
                d3: '//cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min',
                jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
              }
            });
          </script>
          '''
    ))
