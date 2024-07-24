from pyharp import ModelCard, build_endpoint, save_and_return_filepath
from audiotools import AudioSignal

import gradio as gr

import pinecone
from towhee import pipe, ops
import numpy as np

import os

card = ModelCard(
    name='Similarity test',
    description='Test similarity against CLMR embeddings',
    author='Nathan Pruyne',
    tags=['Similarity']
)

"""<YOUR MODEL INITIALIZATION CODE HERE>"""
clmr_pipe = (
    pipe.input('path')
        .map('path', 'frame', ops.audio_decode.ffmpeg())
        .map('frame', 'vecs', ops.audio_embedding.clmr())
        .output('vecs')
)

index_clmr = pinecone.Index(os.environ["PC_API_KEY"], host='clmr-small-index-af8053a.svc.gcp-starter.pinecone.io')

def convert_to_npfloat64(original_array):
    #return np.array(flat_df["flat_vector_embed"][0],dtype=np.float64)
    return np.array(original_array,dtype=np.float64)

def convert_to_npfloat64_to_list(vector_embed_64):
    # list(flat_df["flat_vector_embed_64"][0])
    return list(vector_embed_64)

def flatten_vector_embed(vector_embed):
    return list(vector_embed.flatten())

def process_fn(input_audio_path):
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
    print("Getting embedding")
    embedding = clmr_pipe(input_audio_path).get()[0]
    if len(embedding.shape) == 2:
        embedding = embedding[0]

    clean_embed = convert_to_npfloat64_to_list(convert_to_npfloat64(flatten_vector_embed(embedding)))

    print("Getting match")
    match = index_clmr.query(
        vector=clean_embed,
        top_k=1,
        #include_values=False,
            include_metadata=True
        )['matches'][0]

    print("Match obtained!")
    artists = match.metadata['artists']
    song_title = match.metadata['song']

    if type(artists) is list:
        artists = ' and '.join(artists)

    output_text = f"{song_title} by {artists}"

    """
    <YOUR AUDIO SAVING CODE HERE>
    """
    output_audio_path = save_and_return_filepath(sig)

    return output_audio_path, output_text


# Build the Gradio endpoint
with gr.Blocks() as demo:
    # Define widgets
    inputs = [
        gr.Audio(
            label='Audio Input',
            type='filepath'
        ),
        #<YOUR UI ELEMENTS HERE>
    ]

    # Make an output audio widget
    output = gr.Audio(label='Audio Output', type='filepath')

    output_text = None

    # Add output text widget (OPTIONAL)
    output_text = gr.Textbox(label='Output text')

    # Build the endpoint
    widgets = build_endpoint(inputs, output, process_fn, card, text_out=output_text)

demo.queue()
demo.launch(share=True)
