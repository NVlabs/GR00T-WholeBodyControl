"""Launch a Gradio page for reviewing and trimming LeRobot episodes."""

import argparse
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.data.cleanse_lerobot_dataset import MissingParquetDependency  # noqa: E402
from gear_sonic.data.episode_review import (  # noqa: E402
    EpisodeReference,
    build_episode_catalog,
    compute_joint_velocity,
    find_dataset_paths,
    format_trash_confirmation_message,
    get_parquet_path,
    gradio_allowed_paths,
    joint_names,
    load_episode_table,
    load_info,
    make_joint_velocity_toggle_html,
    move_episode_to_trash,
    navigate_episode_index,
    resolve_episode_video,
    review_sync_head,
    selected_catalog_index_from_table_event,
    trim_episode_to_dataset,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review LeRobot point-block episodes in a local Gradio page.",
    )
    parser.add_argument(
        "--dataset-root",
        default="outputs",
        help="Root containing LeRobot datasets, or a LeRobot dataset directory itself.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset name under --dataset-root, or an absolute dataset path. "
            "Repeat to load multiple datasets. Defaults to all datasets under the root."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/episode_review_trims",
        help="Directory where trimmed one-episode datasets are written.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Request a public Gradio share URL.")
    parser.add_argument(
        "--max-plot-joints",
        type=int,
        default=8,
        help="Number of most-active joints to draw in the velocity plot.",
    )
    return parser.parse_args()


def _require_gradio() -> Any:
    try:
        import gradio as gr  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on operator env
        raise SystemExit(
            "This review page requires Gradio. Install the optional review extras with:\n"
            "  pip install -e 'gear_sonic[review]'\n"
            "or launch with uv, for example:\n"
            "  uv run --with gradio --with pandas --with pyarrow --with matplotlib "
            "python gear_sonic/scripts/review_lerobot_episodes.py"
        ) from exc
    return gr


def _catalog_rows(catalog: list[EpisodeReference]) -> list[list[Any]]:
    return [
        [
            item.label,
            item.dataset_name,
            item.episode_index,
            item.length_frames,
            round(item.duration_s, 3),
            item.prompt,
        ]
        for item in catalog
    ]


def _status_for(item: EpisodeReference, video_path: Path | None) -> str:
    video_status = "missing video"
    if video_path is not None:
        video_status = str(video_path) if video_path.exists() else f"missing video: {video_path}"
    return (
        f"dataset={item.dataset_name} | episode={item.episode_index:06d} | "
        f"frames={item.length_frames} | duration={item.duration_s:.3f}s | {video_status}"
    )


def build_app(
    dataset_paths: list[Path],
    output_root: Path,
    max_plot_joints: int,
) -> Any:
    gr = _require_gradio()
    catalog = build_episode_catalog(dataset_paths)
    if not catalog:
        raise SystemExit("No episodes found in the selected dataset paths.")

    def hide_trash_confirmation():
        return gr.update(visible=False), "", None

    def with_hidden_trash_confirmation(outputs):
        return (*outputs, *hide_trash_confirmation())

    def no_episode_outputs(message: str):
        return (
            0,
            "",
            None,
            "",
            message,
            "",
            0.0,
            0.0,
            message,
        )

    def load_episode_at(index: int):
        if not catalog:
            return no_episode_outputs("No episodes available.")
        index = max(0, min(int(index), len(catalog) - 1))
        item = catalog[index]
        info = load_info(item.dataset_path)
        video_path = resolve_episode_video(item.dataset_path, info, item.episode_index)
        velocity_html = ""
        plot_status = ""
        try:
            table = load_episode_table(get_parquet_path(item.dataset_path, info, item.episode_index))
            timestamps, velocity = compute_joint_velocity(table, item.fps)
            names = joint_names(info)
            velocity_html = make_joint_velocity_toggle_html(
                timestamps,
                velocity,
                names,
                default_limit=max_plot_joints,
                duration_s=item.duration_s,
            )
        except MissingParquetDependency as exc:
            plot_status = str(exc)
        except Exception as exc:  # pragma: no cover - UI diagnostic path
            plot_status = f"Could not render velocity plot: {exc}"

        return (
            index,
            item.label,
            str(video_path) if video_path and video_path.exists() else None,
            item.prompt,
            _status_for(item, video_path),
            velocity_html,
            0.0,
            item.duration_s,
            plot_status,
        )

    def select_from_table(event: gr.SelectData):
        row_index = selected_catalog_index_from_table_event(
            getattr(event, "index", None),
            _catalog_rows(catalog),
            total=len(catalog),
            event_value=getattr(event, "row_value", None) or getattr(event, "value", None),
        )
        return with_hidden_trash_confirmation(load_episode_at(row_index))

    def move_episode(current_index: int, action: str):
        next_index = navigate_episode_index(current_index, action, len(catalog))
        return with_hidden_trash_confirmation(load_episode_at(next_index))

    def trim_current(current_index: int, start_s: float, end_s: float):
        current_index = max(0, min(int(current_index), len(catalog) - 1))
        item = catalog[current_index]
        try:
            result = trim_episode_to_dataset(
                item.dataset_path,
                item.episode_index,
                float(start_s),
                float(end_s),
                output_root=Path(output_root),
            )
        except Exception as exc:
            return f"Trim failed: {exc}", ""

        warning_text = ""
        if result.warnings:
            warning_text = "\nWarnings:\n" + "\n".join(f"- {warning}" for warning in result.warnings)
        status = (
            f"Wrote {result.frames} frames ({result.duration_s:.3f}s) to "
            f"{result.dataset_path}{warning_text}"
        )
        return status, str(result.dataset_path)

    def ask_trash_confirmation(current_index: int):
        if not catalog:
            return gr.update(visible=False), "", None, "No episode to move to trash."
        current_index = max(0, min(int(current_index), len(catalog) - 1))
        return (
            gr.update(visible=True),
            format_trash_confirmation_message(catalog[current_index]),
            current_index,
            "",
        )

    def cancel_trash_confirmation():
        return gr.update(visible=False), "", None, "Trash cancelled."

    def trash_current(current_index: int, reason: str):
        nonlocal catalog
        if not catalog:
            return (
                gr.update(value=[]),
                *no_episode_outputs("No episodes available."),
                "No episode to move to trash.",
            )
        current_index = max(0, min(int(current_index), len(catalog) - 1))
        item = catalog[current_index]
        try:
            result = move_episode_to_trash(
                item.dataset_path,
                item.episode_index,
                reason=reason,
            )
        except Exception as exc:
            return (
                gr.update(value=_catalog_rows(catalog)),
                *load_episode_at(current_index),
                f"Move to trash failed: {exc}",
            )

        catalog = build_episode_catalog(dataset_paths)
        if not catalog:
            status = f"Moved {item.label} to trash: {result.trash_dir}. No episodes remain."
            return (
                gr.update(value=[]),
                *no_episode_outputs("No episodes remain after moving the selected episode to trash."),
                status,
            )
        next_index = min(current_index, len(catalog) - 1)
        status = f"Moved {item.label} to trash: {result.trash_dir}"
        return (
            gr.update(value=_catalog_rows(catalog)),
            *load_episode_at(next_index),
            status,
        )

    def trash_confirmed(confirmed_index: int | None, current_index: int, reason: str):
        delete_index = current_index if confirmed_index is None else confirmed_index
        return (*trash_current(delete_index, reason), *hide_trash_confirmation())

    with gr.Blocks(title="GEAR-SONIC Episode Review") as app:
        gr.Markdown("GEAR-SONIC Episode Review")
        current_index_state = gr.State(0)
        delete_confirm_index_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=4, min_width=360):
                index_table = gr.Dataframe(
                    headers=["label", "dataset", "episode", "frames", "seconds", "prompt"],
                    value=_catalog_rows(catalog),
                    datatype=["str", "str", "number", "number", "number", "str"],
                    interactive=False,
                    label="Data index",
                )
                with gr.Row():
                    first_button = gr.Button("<<", size="sm")
                    previous_button = gr.Button("<", size="sm")
                    next_button = gr.Button(">", size="sm")
                    last_button = gr.Button(">>", size="sm")
                episode_label = gr.Textbox(
                    value=catalog[0].label,
                    label="Episode",
                    interactive=False,
                )
                prompt_box = gr.Textbox(label="Prompt", interactive=False)
                status_box = gr.Textbox(label="Episode status", interactive=False)
                plot_status_box = gr.Textbox(label="Plot status", interactive=False)
            with gr.Column(scale=6, min_width=520):
                video = gr.Video(label="ego_view camera")
                gr.HTML('<div class="episode-video-current">0:00.0</div>')
                velocity_plot = gr.HTML()
                with gr.Row():
                    trim_start = gr.Number(value=0.0, label="Trim start (s)", precision=3)
                    trim_end = gr.Number(value=0.0, label="Trim end (s)", precision=3)
                trim_button = gr.Button("Write trimmed dataset", variant="primary")
                trim_status = gr.Textbox(label="Trim status", interactive=False)
                trim_output = gr.Textbox(label="Trim output path", interactive=False)
                delete_reason = gr.Textbox(label="Trash reason", placeholder="Optional note")
                delete_button = gr.Button("Move episode to trash", variant="stop")
                with gr.Group(visible=False) as delete_confirm_group:
                    delete_confirm_message = gr.Textbox(
                        label="Trash confirmation",
                        interactive=False,
                    )
                    with gr.Row():
                        confirm_delete_yes = gr.Button("Yes", variant="stop", size="sm")
                        confirm_delete_no = gr.Button("No", size="sm")
                delete_status = gr.Textbox(label="Trash status", interactive=False)

        nav_outputs = [
            current_index_state,
            episode_label,
            video,
            prompt_box,
            status_box,
            velocity_plot,
            trim_start,
            trim_end,
            plot_status_box,
        ]
        nav_and_confirmation_outputs = [
            *nav_outputs,
            delete_confirm_group,
            delete_confirm_message,
            delete_confirm_index_state,
        ]
        table_and_nav_outputs = [index_table, *nav_outputs]
        delete_confirmation_outputs = [
            delete_confirm_group,
            delete_confirm_message,
            delete_confirm_index_state,
            delete_status,
        ]
        first_button.click(
            lambda current_index: move_episode(current_index, "first"),
            inputs=[current_index_state],
            outputs=nav_and_confirmation_outputs,
        )
        previous_button.click(
            lambda current_index: move_episode(current_index, "previous"),
            inputs=[current_index_state],
            outputs=nav_and_confirmation_outputs,
        )
        next_button.click(
            lambda current_index: move_episode(current_index, "next"),
            inputs=[current_index_state],
            outputs=nav_and_confirmation_outputs,
        )
        last_button.click(
            lambda current_index: move_episode(current_index, "last"),
            inputs=[current_index_state],
            outputs=nav_and_confirmation_outputs,
        )
        index_table.select(
            select_from_table,
            inputs=None,
            outputs=nav_and_confirmation_outputs,
        )
        trim_button.click(
            trim_current,
            inputs=[current_index_state, trim_start, trim_end],
            outputs=[trim_status, trim_output],
        )
        delete_button.click(
            ask_trash_confirmation,
            inputs=[current_index_state],
            outputs=delete_confirmation_outputs,
        )
        confirm_delete_no.click(
            cancel_trash_confirmation,
            inputs=None,
            outputs=delete_confirmation_outputs,
        )
        confirm_delete_yes.click(
            trash_confirmed,
            inputs=[delete_confirm_index_state, current_index_state, delete_reason],
            outputs=[
                *table_and_nav_outputs,
                delete_status,
                delete_confirm_group,
                delete_confirm_message,
                delete_confirm_index_state,
            ],
        )
        app.load(
            lambda: with_hidden_trash_confirmation(load_episode_at(0)),
            inputs=None,
            outputs=nav_and_confirmation_outputs,
        )
    return app


def main() -> None:
    args = _parse_args()
    dataset_paths = find_dataset_paths(Path(args.dataset_root), args.dataset)
    if not dataset_paths:
        raise SystemExit(f"No LeRobot datasets found under {args.dataset_root}")

    app = build_app(
        dataset_paths=dataset_paths,
        output_root=Path(args.output_root),
        max_plot_joints=args.max_plot_joints,
    )
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        head=review_sync_head(),
        allowed_paths=gradio_allowed_paths(dataset_paths, Path(args.output_root)),
    )


if __name__ == "__main__":
    main()
