from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from backend.models import LATENT_DIM, Discriminator, Generator
from backend.training import generate_images, train_dcgan


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
SAMPLES = OUTPUTS / "samples"
CHECKPOINT = OUTPUTS / "checkpoints" / "latest.pt"

st.set_page_config(page_title="Fashion Lab · DCGAN", page_icon="🧵", layout="wide")

dark_mode = bool(st.session_state.get("dark_mode", False))

palette = {
    "background": "#151719" if dark_mode else "#f6f5f1",
    "surface": "#202428" if dark_mode else "#ffffff",
    "border": "#363c42" if dark_mode else "#e8e5dd",
    "text": "#f0eee8" if dark_mode else "#20211f",
    "muted": "#b6b3aa" if dark_mode else "#686862",
    "label": "#b9b5aa" if dark_mode else "#77756e",
    "accent": "#d0a969" if dark_mode else "#7b6a49",
    "panel_info": "#243244" if dark_mode else "#e8f1fb",
    "panel_info_text": "#d6e4f7" if dark_mode else "#416b9a",
    "chart": "#202428" if dark_mode else "#ffffff",
    "chart_grid": "#363c42" if dark_mode else "#e8e5dd",
}

st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: {"dark" if dark_mode else "light"};
        --app-bg: {palette["background"]};
        --surface: {palette["surface"]};
        --border: {palette["border"]};
        --app-text: {palette["text"]};
        --muted-text: {palette["muted"]};
        --label-text: {palette["label"]};
        --accent: {palette["accent"]};
        --info-bg: {palette["panel_info"]};
        --info-text: {palette["panel_info_text"]};
        --chart-bg: {palette["chart"]};
        --chart-grid: {palette["chart_grid"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--app-bg); color: var(--app-text); }
    [data-testid="stHeader"] { background: var(--app-bg); }
    .block-container { max-width: 1280px; padding-top: 2.3rem; }
    .eyebrow { color: var(--accent); text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; }
    .hero { font-family: Georgia, serif; font-size: 3.1rem; line-height: 1.08; letter-spacing: -.04em; margin: .25rem 0 .75rem; }
    .subtle { color: var(--muted-text); font-size: 1.02rem; }
    .metric-card { background: var(--surface); color: var(--app-text); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem; min-height: 96px; }
    .metric-label { color: var(--label-text); font-size: .76rem; text-transform: uppercase; letter-spacing: .1em; }
    .metric-value { font-size: 1.35rem; font-weight: 650; margin-top: .3rem; }
    div[data-testid="stMetric"] { background: var(--surface); color: var(--app-text); border: 1px solid var(--border); border-radius: 14px; padding: 1rem; }
    div[data-testid="stMetric"] label { color: var(--label-text); }
    .section-title { font-family: Georgia, serif; font-size: 1.7rem; margin: 0 0 .2rem; }
    .stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp li { color: var(--app-text); }
    [data-testid="stCaptionContainer"] p { color: var(--muted-text); }
    [data-baseweb="tab"] { color: var(--muted-text) !important; }
    [data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; }
    [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
    [data-baseweb="select"] > div { background: var(--surface); color: var(--app-text); border-color: var(--border); }
    [role="listbox"], [role="option"] { background: var(--surface); color: var(--app-text); }
    [data-testid="stButton"] > button { background: var(--surface); color: var(--app-text); border-color: var(--border); }
    [data-testid="stButton"] > button[kind="primary"] { background: var(--accent); color: #171411; border-color: var(--accent); }
    [data-testid="stAlert"] { background: var(--info-bg); color: var(--info-text); border-color: var(--border); }
    [data-testid="stVegaLiteChart"] { background: var(--chart-bg); border: 1px solid var(--border); border-radius: 12px; padding: .4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def make_loss_chart(frame: pd.DataFrame) -> alt.Chart:
    """Build a loss chart whose colors follow the selected app theme."""
    chart_data = frame.melt(id_vars="epoch", var_name="network", value_name="loss")
    chart_data["network"] = chart_data["network"].str.replace("_loss", "", regex=False)
    return (
        alt.Chart(chart_data)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("loss:Q", title="Loss"),
            color=alt.Color(
                "network:N",
                title="Network",
                scale=alt.Scale(
                    domain=["generator", "discriminator"],
                    range=["#e7ae5e", "#65a9ee"] if dark_mode else ["#9b7944", "#3c78a8"],
                ),
            ),
            tooltip=[alt.Tooltip("epoch:Q"), alt.Tooltip("network:N"), alt.Tooltip("loss:Q", format=".3f")],
        )
        .properties(height=240, background=palette["surface"])
        .configure_axis(
            labelColor=palette["muted"],
            titleColor=palette["muted"],
            gridColor=palette["border"],
            domainColor=palette["border"],
            tickColor=palette["border"],
        )
        .configure_legend(labelColor=palette["muted"], titleColor=palette["muted"])
        .configure_view(stroke=None)
    )


heading, theme_control = st.columns([5, 1], vertical_alignment="center")
with heading:
    st.markdown('<div class="eyebrow">Generative studio · FashionMNIST</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero">Teach a model to imagine fashion.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Train a Deep Convolutional GAN on 28 × 28 clothing images, then explore how its samples change as training progresses.</div>',
        unsafe_allow_html=True,
    )
with theme_control:
    st.toggle("Dark mode", key="dark_mode", help="Switch the app between light and dark themes.")
st.write("")

generator_parameters = sum(parameter.numel() for parameter in Generator().parameters())
discriminator_parameters = sum(parameter.numel() for parameter in Discriminator().parameters())
saved_epoch = 0
if CHECKPOINT.exists():
    try:
        saved_epoch = int(torch.load(CHECKPOINT, map_location="cpu", weights_only=False).get("epoch", 0))
    except (OSError, RuntimeError, ValueError, KeyError):
        saved_epoch = 0

overview = st.columns(4)
for column, label, value in zip(
    overview,
    ["Dataset", "Image format", "Generator", "Discriminator"],
    ["FashionMNIST · 60k", "28 × 28 · grayscale", f"{generator_parameters:,} params", f"{discriminator_parameters:,} params"],
):
    with column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
train_tab, gallery_tab, about_tab = st.tabs(["Train model", "Image gallery", "How it works"])

with train_tab:
    left, right = st.columns([0.88, 1.12], gap="large")
    with left:
        st.markdown('<div class="section-title">Training controls</div>', unsafe_allow_html=True)
        st.caption("Each epoch passes once through the 60,000 training images.")
        epochs = st.slider("Additional epochs", min_value=1, max_value=30, value=5, step=1)
        batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256], value=128)
        learning_rate = st.select_slider(
            "Learning rate", options=[0.0001, 0.0002, 0.0004], value=0.0002,
            format_func=lambda value: f"{value:.4f}",
        )
        device_options = ["Auto", "CPU"] + (["CUDA"] if torch.cuda.is_available() else [])
        device = st.selectbox("Compute device", device_options, index=0)
        start = st.button("▶  Start training", type="primary", use_container_width=True)
        st.caption("The latest checkpoint resumes automatically. Training stays local to this machine.")

    with right:
        st.markdown('<div class="section-title">Training diagnostics</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0, text=f"Ready · latest saved epoch {saved_epoch}")
        metric_area = st.empty()
        chart_area = st.empty()
        if start:
            status = st.empty()
            try:
                def on_epoch(report: dict) -> None:
                    completed = report["epoch"] - saved_epoch
                    progress_bar.progress(min(completed / epochs, 1.0), text=f"Epoch {report['epoch']} of {saved_epoch + epochs}")
                    metric_area.metric("Latest epoch", f"{report['epoch']}", help=f"Running on {report['device']}")
                    status.image(Image.open(report["sample_path"]), caption="Fixed-noise sample grid · refreshed each epoch", use_container_width=True)
                    if (OUTPUTS / "losses.csv").exists():
                        chart_area.altair_chart(make_loss_chart(pd.read_csv(OUTPUTS / "losses.csv")), use_container_width=True, theme=None)

                with st.spinner("Training the generator and discriminator…"):
                    result = train_dcgan(
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        device_preference=device,
                        progress_callback=on_epoch,
                    )
                st.success(f"Training complete on {result['device']}. Checkpoint saved to {result['checkpoint_path'].relative_to(ROOT)}")
            except Exception as error:
                st.error(f"Training could not finish: {error}")

        losses_path = OUTPUTS / "losses.csv"
        if losses_path.exists():
            frame = pd.read_csv(losses_path)
            if not frame.empty:
                with chart_area:
                    st.altair_chart(make_loss_chart(frame), use_container_width=True, theme=None)
                latest = frame.iloc[-1]
                metric_area.markdown(
                    f"**Latest losses** · G {latest['generator_loss']:.3f} &nbsp; / &nbsp; D {latest['discriminator_loss']:.3f}"
                )
        else:
            metric_area.info("Generator and discriminator losses will appear here after the first epoch.")

with gallery_tab:
    st.markdown('<div class="section-title">Generated clothing</div>', unsafe_allow_html=True)
    st.caption("Epoch previews use the same fixed noise vectors so you can compare training progress.")
    image_count = st.slider("New images to generate", min_value=4, max_value=32, value=16, step=4, key="image_count")
    if st.button("Generate from latest checkpoint", disabled=not CHECKPOINT.exists()):
        try:
            images = generate_images(count=image_count)
            generated_path = SAMPLES / "latest_gallery.png"
            from torchvision.utils import save_image

            generated_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(images, generated_path, nrow=8, normalize=True, value_range=(-1, 1), padding=2)
            st.image(Image.open(generated_path), caption=f"{image_count} new samples from epoch {saved_epoch}", use_container_width=True)
        except Exception as error:
            st.error(f"Image generation failed: {error}")

    epoch_images = sorted(SAMPLES.glob("epoch_*.png"), reverse=True) if SAMPLES.exists() else []
    if epoch_images:
        ordered_images = sorted(epoch_images)
        if len(ordered_images) > 1:
            st.markdown("#### Training progress · same noise, different epoch")
            comparison = st.columns(2)
            with comparison[0]:
                st.image(Image.open(ordered_images[0]), caption=f"Epoch {int(ordered_images[0].stem.split('_')[-1])}", use_container_width=True)
            with comparison[1]:
                st.image(Image.open(ordered_images[-1]), caption=f"Epoch {int(ordered_images[-1].stem.split('_')[-1])}", use_container_width=True)
        st.markdown("#### Recent training snapshots")
        gallery = st.columns(min(3, len(epoch_images)))
        for index, path in enumerate(epoch_images[:6]):
            with gallery[index % len(gallery)]:
                epoch = path.stem.split("_")[-1]
                st.image(Image.open(path), caption=f"Epoch {int(epoch)}", use_container_width=True)
    elif not CHECKPOINT.exists():
        st.info("Train the model to create the first generated image gallery.")

with about_tab:
    st.markdown('<div class="section-title">A two player learning game</div>', unsafe_allow_html=True)
    st.write(
        "The generator turns random 100-dimensional noise into a clothing-like image. "
        "The discriminator sees real FashionMNIST images and generated images, then learns to tell them apart. "
        "The generator learns from the discriminator’s feedback and tries to make more convincing samples."
    )
    st.markdown("**Generator:** dense projection → 7 × 7 feature map → transposed convolutions → `tanh` image")
    st.markdown("**Discriminator:** convolution → downsampling blocks → one real/fake logit")
    st.markdown("**Training objective:** binary cross entropy with logits · Adam · learning rate selected above")
    st.markdown("**Visible evidence:** generator/discriminator loss curves and fixed-noise sample grids by epoch")

st.divider()
st.caption("Fashion Lab · DCGAN · PyTorch · FashionMNIST")
