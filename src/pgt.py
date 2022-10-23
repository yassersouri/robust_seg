"""Variable names:

temporal_size: T
num_classes: NC
num_timestamps: NTS
"""

import dataclasses as dt
import typing as t

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, Optimizer


@dt.dataclass()
class Config:
    optimizer_kwargs: t.Dict[str, float] = dt.field(
        default_factory=lambda: {"lr": 0.03}
    )
    optimization_num_steps: int = 30
    ignore_label_value: int = -100
    plateau_sharpness: float = 0.025

    loss_mul_observation: float = 1.0
    loss_mul_empty: float = 0.7


# noinspection PyPep8Naming
@dt.dataclass
class pGTResult:
    # the pGT labels. Tensor fo shapes, containing the
    # class label for each frame that can be used for training. If a temporal
    # position is skipped by the method the value will be the `ignore_index`
    # which can be specified in the config.
    pgt: Tensor

    # The optional result that can be used for visualization purposes.
    vis_result: t.Optional[t.Dict] = None


def create_output(
    config: Config,
    start_parameters_after_sigmoid: Tensor,
    end_parameters_after_sigmoid: Tensor,
    timestamps: Tensor,
    timestamp_labels: Tensor,
    temporal_size: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Generates the Pseudo Ground Truth from the parameters and the timestamp.

    :param config:
    :param start_parameters_after_sigmoid: Tensor [TNS] float in the range [0, 1]
    :param end_parameters_after_sigmoid: Tensor [TNS] float in the range [0, 1]
    :param timestamps: Tensor [TNS] int
    :param timestamp_labels: Tensor [TNS] long
    :param temporal_size: int
    :param device: device to put the output tensor on
    :return: Tensor [T] long
    """
    (
        absolute_start_values,
        absolute_end_values,
    ) = compute_absolute_left_and_right_of_timestamps(
        start_parameters_after_sigmoid,
        end_parameters_after_sigmoid,
        temporal_size,
        timestamps,
    )

    result = torch.full(
        size=(temporal_size,),
        fill_value=config.ignore_label_value,
        dtype=torch.long,
        device=device,
    )

    for label, start, end in zip(
        timestamp_labels, absolute_start_values, absolute_end_values
    ):
        # rounding the start and end values
        start, end = int(round(start.item())), int(round(end.item()))
        if start == end:
            end += 1
        result[start:end] = label

    return result


def plateau_function(
    temporal_size: int, centers: Tensor, half_widths: Tensor, sharpness: float
) -> Tensor:
    """
    The plateau function.
    centers: Tensor [N], float32
    half_width: Tensor [N], float32

    return [N x temporal_size], float 32, between 0 and 1.
    """
    centers, half_widths = centers.unsqueeze(1), half_widths.unsqueeze(1)
    the_range = torch.arange(
        start=0, end=temporal_size, step=1.0, dtype=torch.float32, device=centers.device
    )
    inside_1 = sharpness * (the_range - centers - half_widths)
    inside_2 = sharpness * (-the_range + centers - half_widths)
    zeros_ = torch.zeros_like(inside_1)
    denom_1_prime = torch.logsumexp(torch.stack((inside_1, zeros_), dim=2), dim=2)
    denom_2_prime = torch.logsumexp(torch.stack((inside_2, zeros_), dim=2), dim=2)
    g_prime = -1 * (denom_1_prime + denom_2_prime)
    g = torch.exp(g_prime)  # [M x T], the plateaus

    return g


def create_windows(
    config: Config,
    start_parameters_after_sigmoid: Tensor,
    end_parameters_after_sigmoid: Tensor,
    timestamps: Tensor,
    temporal_size: int,
) -> Tensor:
    """

    :param config:
    :param start_parameters_after_sigmoid: [NTS], float32, between 0 and 1
    :param end_parameters_after_sigmoid: [NTS], float32, between 0 and 1
    :param timestamps: [NTS], int
    :param temporal_size: int
    :return: # [NTS x T]
    """

    # 0, p1, p2, p3, T
    #    s1, s2, s3
    #    e1, e2, e3

    # left side: (all_points[1:-1] - all_points[0:-2]) * S
    # p1 - 0 * s1
    # p2 - p1 * s2
    # p3 - p2 * s3

    # right side: (all_points[2:] - all_points[1:-1]) * E
    # p2 - p1 * e1
    # p3 - p2 * e2
    # T - p3 * e3

    (
        absolute_left_position,
        absolute_right_position,
    ) = compute_absolute_left_and_right_of_timestamps(
        start_parameters_after_sigmoid=start_parameters_after_sigmoid,
        end_parameters_after_sigmoid=end_parameters_after_sigmoid,
        temporal_size=temporal_size,
        timestamps=timestamps,
    )

    window_centers = (absolute_left_position + absolute_right_position) / 2.0
    window_half_widths = (absolute_right_position - absolute_left_position) / 2.0

    return plateau_function(
        temporal_size=temporal_size,
        centers=window_centers,
        half_widths=window_half_widths,
        sharpness=config.plateau_sharpness,
    )


def compute_absolute_left_and_right_of_timestamps(
    start_parameters_after_sigmoid: Tensor,
    end_parameters_after_sigmoid: Tensor,
    temporal_size: int,
    timestamps: Tensor,
):
    all_points = torch.zeros(
        size=(timestamps.shape[0] + 2,),
        dtype=timestamps.dtype,
        device=timestamps.device,
    )
    all_points[1:-1] = timestamps
    all_points[-1] = temporal_size
    absolute_left_side_length = (
        all_points[1:-1] - all_points[:-2]
    ) * start_parameters_after_sigmoid
    absolute_right_side_length = (
        all_points[2:] - all_points[1:-1]
    ) * end_parameters_after_sigmoid
    absolute_left_position = timestamps - absolute_left_side_length
    absolute_right_position = timestamps + absolute_right_side_length
    return absolute_left_position, absolute_right_position


def overlay_windows(
    windows: Tensor, timestamps_labels: Tensor, num_classes: int
) -> Tensor:
    """
    Let's say we have 4 classes (num_classes = 4).
    And the timestamps labels are: [0, 2, 0] (notice that 1 is not in the labels).
    And the windows are:
    1 1 0 0 0
    0 1 1 0 0
    0 0 0 1 0

    The overlayed output should be:
    1 1 0 1 0
    0 0 0 0 0
    0 1 1 0 0
    0 0 0 0 0

    :param windows: Tensor [NTS x T] float, in the range [0, 1]
    :param timestamps_labels: Tensor [NTS] long
    :param num_classes: int
    :return: Tensor [NC x T] float, in the range [0, 1]
    """
    temporal_size = windows.shape[1]

    out = torch.zeros(
        size=(num_classes, temporal_size),
        dtype=torch.float32,
        device=windows.device,
    )

    out.index_add_(0, timestamps_labels, windows)

    return out


def calculate_empty_loss(windows: Tensor) -> Tensor:
    temporal_max, _ = torch.max(windows, dim=0)  # [T]
    return (1 - temporal_max).sum()  # []


def calculate_loss(
    config: Config,
    nl_probs: Tensor,
    windows: Tensor,
    timestamp_labels: Tensor,
) -> Tensor:
    """
    :param config:
    :param nl_probs: Tensor [NC x T]
    :param windows: Tensor [NTS x T]
    :param timestamp_labels: Tensor [NTS] long
    :return: Tensor with a single value
    """
    windows = overlay_windows(
        windows=windows,
        timestamps_labels=timestamp_labels,
        num_classes=nl_probs.shape[0],
    )  # [NC x T]

    observation_loss = (windows * nl_probs).sum()  # []
    empty_loss = calculate_empty_loss(windows)  # []

    return (
        observation_loss * config.loss_mul_observation
        + empty_loss * config.loss_mul_empty
    )


def generate_baseline_pgt(
    timestamps: Tensor,
    labels: np.ndarray,
    intermediate_features: Tensor,
    ignore_label: int = -100,
) -> Tensor:
    boundary_target = np.ones(labels.shape) * ignore_label
    boundary_target[: timestamps[0]] = labels[
        timestamps[0]
    ]  # frames before first single frame has same label
    left_bound = [0]

    # Forward to find action boundaries
    for i in range(len(timestamps) - 1):
        start = timestamps[i]
        end = timestamps[i + 1] + 1
        left_score = torch.zeros(end - start - 1, dtype=torch.float)
        for t in range(start + 1, end):
            center_left = torch.mean(
                intermediate_features[:, left_bound[-1] : t], dim=1
            )
            diff_left = intermediate_features[:, start:t] - center_left.reshape(-1, 1)
            score_left = torch.mean(torch.norm(diff_left, dim=0))

            center_right = torch.mean(intermediate_features[:, t:end], dim=1)
            diff_right = intermediate_features[:, t:end] - center_right.reshape(-1, 1)
            score_right = torch.mean(torch.norm(diff_right, dim=0))

            left_score[t - start - 1] = (
                (t - start) * score_left + (end - t) * score_right
            ) / (end - start)

        cur_bound = torch.argmin(left_score) + start + 1
        left_bound.append(cur_bound.item())

    # Backward to find action boundaries
    right_bound = [labels.shape[0]]
    for i in range(len(timestamps) - 1, 0, -1):
        start = timestamps[i - 1]
        end = timestamps[i] + 1
        right_score = torch.zeros(end - start - 1, dtype=torch.float)
        for t in range(end - 1, start, -1):
            center_left = torch.mean(intermediate_features[:, start:t], dim=1)
            diff_left = intermediate_features[:, start:t] - center_left.reshape(-1, 1)
            score_left = torch.mean(torch.norm(diff_left, dim=0))

            center_right = torch.mean(
                intermediate_features[:, t : right_bound[-1]], dim=1
            )
            diff_right = intermediate_features[:, t:end] - center_right.reshape(-1, 1)
            score_right = torch.mean(torch.norm(diff_right, dim=0))

            right_score[t - start - 1] = (
                (t - start) * score_left + (end - t) * score_right
            ) / (end - start)

        cur_bound = torch.argmin(right_score) + start + 1
        right_bound.append(cur_bound.item())

    # Average two action boundaries for same segment and generate pseudo labels
    left_bound = left_bound[1:]
    right_bound = right_bound[1:]
    num_bound = len(left_bound)
    for i in range(num_bound):
        temp_left = left_bound[i]
        temp_right = right_bound[num_bound - i - 1]
        middle_bound = int((temp_left + temp_right) / 2)
        boundary_target[timestamps[i] : middle_bound] = labels[timestamps[i]]
        boundary_target[middle_bound : timestamps[i + 1] + 1] = labels[
            timestamps[i + 1]
        ]

    boundary_target[timestamps[-1] :] = labels[
        timestamps[-1]
    ]  # frames after last single frame has same label

    return torch.from_numpy(boundary_target)


def generate_oracle_pgt(
    timestamps: t.List[int], labels: t.List[int], ignore_label: int = -100
) -> Tensor:
    """
    :param timestamps: list of length [NTS] containing the time indices of timestamps
    :param labels: list of length [T] containing the ground truth labels
    :param ignore_label: the ignore label
    :return: Tensor of shape [T] of dtype long containing the oracle_pgt
    """
    oracle_pgt = torch.ones(len(labels), dtype=torch.long) * ignore_label

    aug_ts = [-1]
    for ts in timestamps:
        aug_ts.append(ts)
    aug_ts.append(len(labels))
    for i in range(1, len(aug_ts) - 1):
        oracle_pgt[aug_ts[i]] = labels[aug_ts[i]]
        for l_idx in range(aug_ts[i], aug_ts[i - 1], -1):
            if labels[l_idx] == labels[aug_ts[i]]:
                oracle_pgt[l_idx] = labels[aug_ts[i]]
            else:
                break
        for r_idx in range(aug_ts[i], aug_ts[i + 1]):
            if labels[r_idx] == labels[aug_ts[i]]:
                oracle_pgt[r_idx] = labels[aug_ts[i]]
            else:
                break
    return oracle_pgt


def generate_pgt_hard(
    nl_probs: Tensor,
    timestamps: Tensor,
    timestamp_labels: Tensor,
    config: Config = Config(),
    visualization_result: bool = False,
) -> pGTResult:

    temporal_size = nl_probs.shape[1]

    # creating the start and end parameters, and gap
    # FIXME: randomness
    values = torch.zeros((timestamps.shape[0] + 1, 3), device=nl_probs.device)
    values[0, 0] = float("-inf")
    values[-1, 1] = float("-inf")

    parameters = torch.nn.Parameter(values, requires_grad=True)

    # creating the optimizer
    optimizer: Optimizer = Adam(params=[parameters], **config.optimizer_kwargs)

    if visualization_result:
        vis_result = {
            "total_loss_values": [],
            "windows": [],
            "start_values": [],
            "end_values": [],
        }
    else:
        vis_result = None

    start_parameters_after_softmax = F.softmax(parameters, dim=1)[:-1, 1]
    end_parameters_after_softmax = F.softmax(parameters, dim=1)[1:, 0]

    for num_iter in range(config.optimization_num_steps):
        windows = create_windows(
            config,
            start_parameters_after_softmax,
            end_parameters_after_softmax,
            timestamps,
            temporal_size,
        )  # [NTS x T]

        loss = calculate_loss(
            config,
            nl_probs,
            windows,
            timestamp_labels,
        )

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        if visualization_result:
            vis_result["total_loss_values"].append(loss.item())
            vis_result["windows"].append(windows.clone().detach())
            vis_result["start_values"].append(
                start_parameters_after_softmax.clone().detach()
            )
            vis_result["end_values"].append(
                end_parameters_after_softmax.clone().detach()
            )

        start_parameters_after_softmax = F.softmax(parameters, dim=1)[:-1, 1]
        end_parameters_after_softmax = F.softmax(parameters, dim=1)[1:, 0]

    pgt = create_output(
        config=config,
        start_parameters_after_sigmoid=start_parameters_after_softmax,
        end_parameters_after_sigmoid=end_parameters_after_softmax,
        timestamps=timestamps,
        timestamp_labels=timestamp_labels,
        temporal_size=temporal_size,
    )

    return pGTResult(pgt=pgt, vis_result=vis_result)
