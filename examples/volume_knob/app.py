from pyharp import ModelCard, build_endpoint, save_and_return_filepath
from audiotools import AudioSignal

import gradio as gr


card = ModelCard(
    name='Volume Knob',
    description='Example Volume Knob',
    author='Nathan Pruyne',
    tags=['example', 'volume']
)

"""<YOUR MODEL INITIALIZATION CODE HERE>"""


def process_fn(input_audio_path, volume_amount):
    """
    This function defines the audio processing steps

    Args:
        input_audio_path (str): the audio filepath to be processed.

        <YOUR_KWARGS>: additional keyword arguments necessary for processing.
            NOTE: These should correspond to and match order of UI elements defined below.

    Returns:
        output_audio_path (str): the filepath of the processed audio.
    """

    """
    <YOUR AUDIO LOADING CODE HERE>
    """
    sig = AudioSignal(input_audio_path)

    """
    <YOUR AUDIO PROCESSING CODE HERE>
    """
    sig.audio_data = volume_amount * sig.audio_data

    """
    <YOUR AUDIO SAVING CODE HERE>
    """
    output_audio_path = save_and_return_filepath(sig)

    return output_audio_path


# Build the Gradio endpoint
with gr.Blocks() as demo:
    # Define widgets
    inputs = [
        gr.Audio(
            label='Audio Input',
            type='filepath'
        ),
        gr.Slider(
            minimum = 0.0,
            maximum = 1.0,
            step = 0.05,
            value = 0.5,
            label = "Volume (percentage)"
        )
    ]

    # Make an output audio widget
    output = gr.Audio(label='Audio Output', type='filepath')

    # Build the endpoint
    widgets = build_endpoint(inputs, output, process_fn, card)

demo.queue()
demo.launch(share=True)
