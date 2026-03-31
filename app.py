import os
import gradio as gr
import asyncio
from dotenv import load_dotenv
from pipeline import NotetakingPipeline

load_dotenv()


async def transcribe_generator(audio_path):
    if not audio_path:
        yield "No audio provided.", "Please upload or record audio first.", "Please upload or record audio first."
        return

    print(f"Processing audio: {audio_path}")
    pipeline = NotetakingPipeline()

    # Run the pipeline in the background
    pipeline_task = asyncio.create_task(pipeline.process_file(audio_path))

    # While the pipeline is running, yield snapshots of the state
    while not pipeline_task.done():
        transcript = pipeline.state.get_transcript_snapshot()
        summary = pipeline.state.get_summary_snapshot()
        entities = pipeline.state.get_entity_snapshot()

        yield transcript, summary, entities
        await asyncio.sleep(0.5)  # Refresh rate for UI

    # Wait for any final results
    await pipeline_task

    # Final yield
    yield (
        pipeline.state.get_transcript_snapshot(),
        pipeline.state.get_summary_snapshot(),
        pipeline.state.get_entity_snapshot(),
    )


# Create Gradio UI with custom dark mode aesthetic
css = """
body {
    background-color: #0b0f19;
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}
.gradio-container {
    max-width: 1280px !important;
}
textarea {
    background-color: #1f2937 !important;
    border: 1px solid #374151 !important;
    color: #f3f4f6 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
h1 {
    color: #60a5fa !important;
    text-align: center;
    font-weight: 800;
}
.btn-primary {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39) !important;
}
.btn-primary:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    transform: translateY(-1px);
}
"""

with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🎙️ Qwen Notetaker: Streaming Async Pipeline")
    gr.Markdown(
        "Record or upload audio to generate smart notes using standalone Qwen3-Omni ASR and Silero VAD."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath", label="Upload or Record File", value=None, format="wav"
            )
            submit_btn = gr.Button(
                "Generate Notes", variant="primary", elem_classes="btn-primary"
            )

        with gr.Column(scale=1):
            transcription_box = gr.Textbox(
                label="Transcription (Timestamped)",
                lines=10,
                placeholder="Transcription will appear here...",
            )
            summary_box = gr.Textbox(
                label="Incremental Summaries",
                lines=10,
                placeholder="Summaries will appear here...",
            )
            entities_box = gr.Textbox(
                label="Extracted Entities",
                lines=10,
                placeholder="Entities will appear here...",
            )

    submit_btn.click(
        fn=transcribe_generator,
        inputs=[audio_input],
        outputs=[transcription_box, summary_box, entities_box],
    )

if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY") and not os.environ.get("QWEN_API_KEY"):
        print("Warning: API Keys not set in environment.")
    demo.launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", 7860)),
    )
