"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        random_acc: bool = False,
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if random_acc: # ceter_fractions = [0.16, 0.08, 0.04, 0.02], accelerations = [1, 16]
            assert len(accelerations) == 2 and len(center_fractions) == 0, \
                f"If random_acc is {random_acc}, the length of 'center_fractions' must be 0 and the length of 'accelerations' must be 2!"
        else:
            if isinstance(center_fractions, float):
                center_fractions = [center_fractions]
            if isinstance(accelerations, int):
                accelerations = [accelerations]
            if len(center_fractions) != len(accelerations) and not allow_any_combination:
                raise ValueError(
                    "Number of center fractions should match number of accelerations "
                    "if random_acc is False."
                )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.random_acc = random_acc
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.random_acc:
            accelerate = self.rng.uniform(self.accelerations[0], self.accelerations[1])
            center_fraction = 0.32 / accelerate
            return center_fraction, accelerate
        else:
            if self.allow_any_combination:
                return self.rng.choice(self.center_fractions), self.rng.choice(
                    self.accelerations
                )
            else:
                choice = self.rng.randint(len(self.center_fractions))
                return self.center_fractions[choice], self.accelerations[choice]


class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space dataset. If the
    k-space dataset has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return self.rng.uniform(size=num_cols) < prob


class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample dataset with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with approximate acceleration matching.

    The mask selects a subset of columns from the input k-space dataset. If the
    k-space dataset has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil dataset.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_cols)
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask


class MagicMaskFunc(MaskFunc):
    """
    Masking function for exploiting conjugate symmetry via offset-sampling.

    This function applies the mask described in the following paper:

    Defazio, A. (2019). Offset Sampling Improves Deep Learning based
    Accelerated MRI Reconstructions by Exploiting Symmetry. arXiv preprint,
    arXiv:1912.01101.

    It is essentially an equispaced mask with an offset for the opposite site
    of k-space. Since MRI images often exhibit approximate conjugate k-space
    symmetry, this mask is generally more efficient than a standard equispaced
    mask.

    Similarly to ``EquispacedMaskFunc``, this mask will usually undereshoot the
    target acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=acceleration)

        if offset % 2 == 0:
            offset_pos = offset + 1
            offset_neg = offset + 2
        else:
            offset_pos = offset - 1 + 3
            offset_neg = offset - 1 + 0

        poslen = (num_cols + 1) // 2
        neglen = num_cols - (num_cols + 1) // 2
        mask_positive = np.zeros(poslen, dtype=np.float32)
        mask_negative = np.zeros(neglen, dtype=np.float32)

        mask_positive[offset_pos::acceleration] = 1
        mask_negative[offset_neg::acceleration] = 1
        mask_negative = np.flip(mask_negative)

        mask = np.concatenate((mask_positive, mask_negative))

        return np.fft.fftshift(mask)  # shift mask and return


class MagicMaskFractionFunc(MagicMaskFunc):
    """
    Masking function for exploiting conjugate symmetry via offset-sampling.

    This function applies the mask described in the following paper:

    Defazio, A. (2019). Offset Sampling Improves Deep Learning based
    Accelerated MRI Reconstructions by Exploiting Symmetry. arXiv preprint,
    arXiv:1912.01101.

    It is essentially an equispaced mask with an offset for the opposite site
    of k-space. Since MRI images often exhibit approximate conjugate k-space
    symmetry, this mask is generally more efficient than a standard equispaced
    mask.

    Similarly to ``EquispacedMaskFractionFunc``, this method exactly matches
    the target acceleration by adjusting the offsets.
    """

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        fraction_low_freqs, acceleration = self.choose_acceleration()
        num_cols = shape[-2]
        num_low_frequencies = round(num_cols * fraction_low_freqs)

        # bound the number of low frequencies between 1 and target columns
        target_columns_to_sample = round(num_cols / acceleration)
        num_low_frequencies = max(min(num_low_frequencies, target_columns_to_sample), 1)

        # adjust acceleration rate based on target acceleration.
        adjusted_target_columns_to_sample = (
            target_columns_to_sample - num_low_frequencies
        )
        adjusted_acceleration = 0
        if adjusted_target_columns_to_sample > 0:
            adjusted_acceleration = round(num_cols / adjusted_target_columns_to_sample)

        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        accel_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, adjusted_acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, accel_mask, num_low_frequencies


class VD1DMaskFunc(MaskFunc):
    """
    1D Variable-Density 掩码（相位编码方向）。
    复用既有 vd_mask_1d() 完成“中心+外圈”的一次性采样，
    然后拆分为 center_mask 与 acceleration_mask 返回。
    """
    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        random_acc: bool = False,
        scheme: str = "exp",                 # 传给 vd_mask_1d 的 scheme
    ):
        super().__init__(center_fractions, accelerations, random_acc)
        self.scheme = scheme

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        直接在这里完成整套采样，然后拆分为 center / accel。
        """
        # 1) 基类选择一组 (center_fraction, acceleration)
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)

        # 2) 用内部 rng 生成一个种子传给 vd_mask_1d（保证可复现）
        #    注意：__call__ 已经用 temp_seed 固定了 self.rng 的状态
        seed_val = int(self.rng.randint(0, 2**31 - 1))

        # 3) 生成“整体” 1D 掩码（包含中心 + 外圈），长度=num_cols
        full_mask_np = vd_mask_1d(
            n_pe=num_cols,
            AF=int(acceleration),
            center_fraction=float(center_fraction),
            scheme=self.scheme,
            seed=seed_val,
        ).astype(np.float32)

        # 4) 用基类的中心构造逻辑生成“中心掩码” 1D
        center_mask_np = self.calculate_center_mask(shape, num_low_frequencies).astype(np.float32).reshape(-1)

        # 5) “外圈掩码” = full_mask - center_mask（注意不能为负）
        accel_mask_np = (full_mask_np > 0.5).astype(np.float32) - (center_mask_np > 0.5).astype(np.float32)
        accel_mask_np = (accel_mask_np > 0.5).astype(np.float32)

        # 6) reshape 成 fastMRI 需要的张量形状
        center_mask = self.reshape_mask(center_mask_np, shape)      # torch.float32
        acceleration_mask = self.reshape_mask(accel_mask_np, shape) # torch.float32

        return center_mask, acceleration_mask, num_low_frequencies


def vd_mask_1d(
    n_pe: int,
    AF: float = 8.0,
    center_fraction: float = 0.04,  # ✅ 新参数：直接控制 ACS 比例
    scheme: str = "exp",
    seed: int = None,
) -> np.ndarray:
    """
    生成一维可变密度(VD)采样掩码 (沿相位编码方向)
    - n_pe: 相位编码 Lines 数
    - AF: 目标加速因子 (>1)
    - center_fraction: 中心 fully-sampled 区域比例 (例如 0.08 表示 8%)
    - scheme: "poly" 或 "exp"
    - seed: 随机种子
    返回: mask (shape: [n_pe], dtype=np.uint8)
    """
    # 由 center_fraction 决定 ACS 条数
    acs = int(n_pe * center_fraction)

    if AF <= 4:
        a, b = 7, 3 # 10, 1.8, 4 #4x
    elif AF == 6:
        a, b = 11, 3  # 16, 1.8, 8 #8x
    elif AF==8:
        a, b = 16, 3  # 16, 1.8, 8 #8x
    elif AF==12:
        a, b = 16, 3  # 16, 1.8, 8 #8x
    elif AF==16:
        a, b = 20, 3  # 16, 1.8, 8 #8x
    else:
        a, b = 7, 3

    # acs =0
    assert AF > 1 and 0 <= acs <= n_pe
    rng = np.random.default_rng(seed)

    k = np.arange(n_pe) - (n_pe - 1) / 2.0
    r = np.abs(k) / (n_pe / 2.0)  # 归一化半径, 0..1

    if scheme == "poly":
        alpha=1.0 # 有需要再指定
        p = (1.0 - r) ** alpha
    elif scheme == "exp":
        # a 控制密度衰减尺度(0~1)，b 控制陡峭度
        p = np.exp(- (a * (r ** b)))
    else:
        raise ValueError("scheme must be 'poly' or 'exp'")

    p = p / p.max()  # 归一化到 [0,1]

    # 目标总采样数 = n_pe / AF；其中 ACS 必须全采
    target = int(round(n_pe / AF))
    acs = min(acs, target)  # 防止 ACS 大于目标采样数
    mask = np.zeros(n_pe, dtype=bool)

    # 放置 ACS（居中）
    if acs > 0:
        c0 = (n_pe - acs) // 2
        mask[c0:c0+acs] = True

    # 剩余 slots
    left = target - mask.sum()
    if left <= 0:
        return mask

    # 非 ACS 区域按概率抽样
    candidates = np.where(~mask)[0]
    prob = p[candidates]
    prob = prob / prob.sum()  # 作为多项式/指数权重
    # 无放回按权重选择
    choose = rng.choice(candidates, size=left, replace=False, p=prob)
    mask[choose] = True

    # 若想更精确逼近 AF，可在此处做微调（例如：若超采样，随机剔除边缘概率最低的点；若欠采样，再按概率补点）
    # 这里简单保证总数=target
    if mask.sum() > target:
        extra = mask.sum() - target
        idx = np.where(mask & (~(np.arange(n_pe) >= (n_pe-acs)//2) | ~(np.arange(n_pe) < (n_pe+acs)//2)))[0]
        drop = rng.choice(idx, size=extra, replace=False)
        mask[drop] = False
    elif mask.sum() < target:
        need = target - mask.sum()
        idx = np.where(~mask)[0]
        add = rng.choice(idx, size=need, replace=False)
        mask[add] = True

    return mask.astype(np.uint8)

def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
    random_acc: bool = False,
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.

    Returns:
        A mask func for the target mask type.
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations, random_acc)
    elif mask_type_str == "equispaced":
        return EquiSpacedMaskFunc(center_fractions, accelerations, random_acc)
    elif mask_type_str == "equispaced_fraction":
        return EquispacedMaskFractionFunc(center_fractions, accelerations, random_acc)
    elif mask_type_str == "magic":
        return MagicMaskFunc(center_fractions, accelerations, random_acc)
    elif mask_type_str == "magic_fraction":
        return MagicMaskFractionFunc(center_fractions, accelerations, random_acc)
    elif mask_type_str == "vd":   # 新增
        return VD1DMaskFunc(center_fractions, accelerations, random_acc)
    else:
        raise ValueError(f"{mask_type_str} not supported")
