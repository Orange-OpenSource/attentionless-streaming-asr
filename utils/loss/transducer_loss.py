# Software Name: attentionless-streaming-asr
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html 

import logging
import math
import warnings

import torch
from torch.autograd import Function
from torch.nn import Module

NUMBA_VERBOSE = 0

logger = logging.getLogger(__name__)

try:
    from numba import cuda

    # Numba is extra verbose and this may lead to log.txt file of multiple gigabytes... we deactivate
    if not NUMBA_VERBOSE:
        logger.info(
            "Numba verbose is deactivated. To enable it, set NUMBA_VERBOSE to 1."
        )

        nb_logger = logging.getLogger("numba")
        nb_logger.setLevel(logging.ERROR)  # only show error

        from numba.core.errors import NumbaPerformanceWarning

        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
    else:
        logger.info(
            "Numba verbose is enabled. To deactivate it, set NUMBA_VERBOSE to 0."
        )

except ImportError:
    err_msg = "The optional dependency Numba is needed to use this module\n"
    err_msg += "Cannot import numba. To use Transducer loss\n"
    err_msg += "Please follow the instructions below\n"
    err_msg += "=============================\n"
    err_msg += "If you use your localhost:\n"
    err_msg += "pip install numba\n"
    err_msg += "export NUMBAPRO_LIBDEVICE='/usr/local/cuda/nvvm/libdevice/' \n"
    err_msg += "export NUMBAPRO_NVVM='/usr/local/cuda/nvvm/lib64/libnvvm.so' \n"
    err_msg += "================================ \n"
    err_msg += "If you use conda:\n"
    err_msg += "conda install numba cudatoolkit"
    raise ImportError(err_msg)


@cuda.jit()
def cu_kernel_forward(log_probs, labels, alpha, log_p, T, U, blank, lock):
    """
    Compute forward pass for the forward-backward algorithm using Numba cuda kernel.
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf

    Arguments
    ---------
    log_probs : torch.Tensor
        4D Tensor of (batch x TimeLength x LabelLength x outputDim) from the Transducer network.
    labels : torch.Tensor
        2D Tensor of (batch x MaxSeqLabelLength) containing targets of the batch with zero padding.
    alpha : torch.Tensor
        3D Tensor of (batch x TimeLength x LabelLength) for forward computation.
    log_p : torch.Tensor
        1D Tensor of (batch) for forward cost computation.
    T : torch.Tensor
        1D Tensor of (batch) containing TimeLength of each target.
    U : torch.Tensor
        1D Tensor of (batch) containing LabelLength of each target.
    blank : int
        Blank index.
    lock : torch.Tensor
        2D Tensor of (batch x LabelLength) containing bool(1-0) lock for parallel computation.
    """

    # parallelize the forward algorithm over batch and target length dim
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x
    t = 0
    if u <= U[b]:
        # for each (B,U) Thread
        # wait the unlock of the previous computation of Alpha[b,U-1,:]
        # Do the computation over the whole Time sequence on alpha[B,U,:]
        # and then unlock the target U+1 for computation
        while t < T[b]:
            if u == 0:
                if t > 0:
                    alpha[b, t, 0] = (
                        alpha[b, t - 1, 0] + log_probs[b, t - 1, 0, blank]
                    )
                cuda.atomic.add(lock, (b, u + 1), -1)
                t += 1
            else:
                if cuda.atomic.add(lock, (b, u), 0) < 0:
                    if t == 0:
                        alpha[b, 0, u] = (
                            alpha[b, 0, u - 1]
                            + log_probs[b, 0, u - 1, labels[b, u - 1]]
                        )
                    else:
                        # compute emission prob
                        emit = (
                            alpha[b, t, u - 1]
                            + log_probs[b, t, u - 1, labels[b, u - 1]]
                        )
                        # compute no_emission prob
                        no_emit = (
                            alpha[b, t - 1, u] + log_probs[b, t - 1, u, blank]
                        )
                        # do logsumexp between log_emit and log_no_emit
                        alpha[b, t, u] = max(no_emit, emit) + math.log1p(
                            math.exp(-abs(no_emit - emit))
                        )
                    if u < U[b]:
                        cuda.atomic.add(lock, (b, u + 1), -1)
                    cuda.atomic.add(lock, (b, u), 1)
                    t += 1
        if u == U[b]:
            # for each thread b (utterance)
            # normalize the loss over time
            log_p[b] = (
                alpha[b, T[b] - 1, U[b]] + log_probs[b, T[b] - 1, U[b], blank]
            ) / T[b]


@cuda.jit()
def cu_kernel_backward(log_probs, labels, beta, log_p, T, U, blank, lock):
    """
    Compute backward pass for the forward-backward algorithm using Numba cuda kernel.
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf

    Arguments
    ---------
    log_probs : torch.Tensor
        4D Tensor of (batch x TimeLength x LabelLength x outputDim) from the Transducer network.
    labels : torch.Tensor
        2D Tensor of (batch x MaxSeqLabelLength) containing targets of the batch with zero padding.
    beta : torch.Tensor
        3D Tensor of (batch x TimeLength x LabelLength) for backward computation.
    log_p : torch.Tensor
        1D Tensor of (batch) for backward cost computation.
    T : torch.Tensor
        1D Tensor of (batch) containing TimeLength of each target.
    U : torch.Tensor
        1D Tensor of (batch) containing LabelLength of each target.
    blank : int
        Blank index.
    lock : torch.Tensor
        2D Tensor of (batch x LabelLength) containing bool(1-0) lock for parallel computation.
    """
    # parallelize the forward algorithm over batch and target length dim
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x
    t = T[b] - 1
    if u <= U[b]:
        # for each (B,U) Thread
        # wait the unlock of the next computation of beta[b,U+1,:]
        # Do the computation over the whole Time sequence on beta[B,U,:]
        # and then unlock the target U-1 for computation
        while t >= 0:
            if u == U[b]:
                if t == T[b] - 1:
                    beta[b, t, u] = log_probs[b, t, u, blank]
                else:
                    beta[b, t, u] = (
                        beta[b, t + 1, u] + log_probs[b, t, u, blank]
                    )
                cuda.atomic.add(lock, (b, u - 1), -1)
                t -= 1
            else:
                if cuda.atomic.add(lock, (b, u), 0) < 0:
                    if t == T[b] - 1:
                        # do logsumexp between log_emit and log_no_emit
                        beta[b, t, u] = (
                            beta[b, t, u + 1] + log_probs[b, t, u, labels[b, u]]
                        )
                    else:
                        # compute emission prob
                        emit = (
                            beta[b, t, u + 1] + log_probs[b, t, u, labels[b, u]]
                        )
                        # compute no_emission prob
                        no_emit = beta[b, t + 1, u] + log_probs[b, t, u, blank]
                        # do logsumexp between log_emit and log_no_emit
                        beta[b, t, u] = max(no_emit, emit) + math.log1p(
                            math.exp(-abs(no_emit - emit))
                        )
                    if u > 0:
                        cuda.atomic.add(lock, (b, u - 1), -1)
                    cuda.atomic.add(lock, (b, u), 1)
                    t -= 1
    if u == 0:
        # for each thread b (utterance)
        # normalize the loss over time
        log_p[b] = beta[b, 0, 0] / T[b]


@cuda.jit()
def cu_kernel_compute_grad(log_probs, labels, alpha, beta, grads, T, U, blank,c_fast_emit):
    """
    Compute gradient for the forward-backward algorithm using Numba cuda kernel.
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf

    Arguments
    ---------
    log_probs : torch.Tensor
        4D Tensor of (batch x TimeLength x LabelLength x outputDim) from the Transducer network.
    labels : torch.Tensor
        2D Tensor of (batch x MaxSeqLabelLength) containing targets of the batch with zero padding.
    alpha : torch.Tensor
        3D Tensor of (batch x TimeLength x LabelLength) for backward computation.
    beta : torch.Tensor
        3D Tensor of (batch x TimeLength x LabelLength) for backward computation.
    grads : torch.Tensor
        Grads for backward computation.
    T : torch.Tensor
        1D Tensor of (batch) containing TimeLength of each target.
    U : torch.Tensor
        1D Tensor of (batch) containing LabelLength of each target.
    blank : int
        Blank index.
    """
    # parallelize the gradient computation over batch and timeseq length dim
    t = cuda.blockIdx.x
    b = cuda.threadIdx.x
    if t < T[b]:
        # compute the gradient for no_emit prob
        if t == 0:
            grads[b, T[b] - 1, U[b], blank] = -math.exp(
                alpha[b, T[b] - 1, U[b]]
                + log_probs[b, T[b] - 1, U[b], blank]
                - beta[b, 0, 0]
            )

        if t < T[b] - 1:
            for u in range(U[b] + 1):
                grads[b, t, u, blank] = alpha[b, t, u] + beta[b, t + 1, u]
                grads[b, t, u, blank] = -math.exp(
                    grads[b, t, u, blank]
                    + log_probs[b, t, u, blank]
                    - beta[b, 0, 0]
                )
        # compute the gradient for emit prob
        for u, l in enumerate(labels[b]):
            if u < U[b]:
                grads[b, t, u, l] = alpha[b, t, u] + beta[b, t, u + 1]
                grads[b, t, u, l] = -(math.exp(math.log1p(c_fast_emit)))*math.exp(
                    grads[b, t, u, l] + log_probs[b, t, u, l] - beta[b, 0, 0]
                )


class Transducer(Function):
    """
    This class implements the Transducer loss computation with forward-backward algorithm
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf

    This class use torch.autograd.Function. In fact of using the forward-backward algorithm,
    we need to compute the gradient manually.

    This class can't be instantiated, please refer to TransducerLoss class

    It is also possible to use this class directly by using Transducer.apply
    """

    @staticmethod
    def forward(ctx, log_probs, labels, T, U, blank, reduction,c_fast_emit):
        """Computes the transducer loss."""
        log_probs = log_probs.detach()
        B, maxT, maxU, A = log_probs.shape
        grads = torch.zeros(
            (B, maxT, maxU, A), dtype=log_probs.dtype, device=log_probs.device
        )
        alpha = torch.zeros(
            (B, maxT, maxU), device=log_probs.device, dtype=log_probs.dtype
        )
        beta = torch.zeros(
            (B, maxT, maxU), device=log_probs.device, dtype=log_probs.dtype
        )
        lock = torch.zeros(
            (B, maxU), dtype=torch.int32, device=log_probs.device
        )
        log_p_alpha = torch.zeros(
            (B,), device=log_probs.device, dtype=log_probs.dtype
        )
        log_p_beta = torch.zeros(
            (B,), device=log_probs.device, dtype=log_probs.dtype
        )
        cu_kernel_forward[B, maxU](
            log_probs, labels, alpha, log_p_alpha, T, U, blank, lock
        )
        lock = lock * 0
        cu_kernel_backward[B, maxU](
            log_probs, labels, beta, log_p_beta, T, U, blank, lock
        )
        cu_kernel_compute_grad[maxT, B](
            log_probs, labels, alpha, beta, grads, T, U, blank, c_fast_emit
        )
        ctx.grads = grads
        del alpha, beta, lock, log_p_beta, T, U, log_probs, labels
        torch.cuda.empty_cache()
        if reduction == "mean":
            return -log_p_alpha.mean()
        elif reduction == "sum":
            return sum(-log_p_alpha)
        elif reduction == "none":
            return -log_p_alpha
        else:
            raise Exception("Unexpected reduction {}".format(reduction))

    @staticmethod
    def backward(ctx, grad_output):
        """Backward computations for the transducer loss."""
        grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul_(grad_output), None, None, None, None, None, None

# #region Pruned Fast RNN-T

# def save_and_plot_tot_grad(
#     px_grad: torch.Tensor,
#     py_grad: torch.Tensor,
#     ids: List[str],
#     x_lens: torch.Tensor,
#     y_lens: torch.Tensor,
#     plot_dir: str,
# ):
#     """Save and plot the tot_grad.
#     Args:
#       px_grad:
#         A tensor of shape (B, S, T+1). It contains the fake gradient.
#       py_grad:
#         A tensor of shape (B, S+1, T). It contains the fake gradient.
#       x_lens:
#         A 1-D tensor of shape (B,), specifying the number of valid acoustic
#         frames in tot_grad for each utterance in the batch.
#       y_lens:
#         A 1-D tensor of shape (B,), specifying the number of valid tokens
#         in tot_grad for each utterance in the batch.
#       plot_dir: a path where the plot will be saved.
#     """
#     import os
#     import matplotlib.pyplot as plt

#     B = px_grad.size(0)  # batch_size
#     S = px_grad.size(1)  # text_length
#     T = px_grad.size(2) - 1  # audio_length
#     px_grad_pad = torch.zeros(
#         (B, 1, T + 1), dtype=px_grad.dtype, device=px_grad.device
#     )
#     py_grad_pad = torch.zeros(
#         (B, S + 1, 1), dtype=py_grad.dtype, device=py_grad.device
#     )

#     px_grad_padded = torch.cat([px_grad, px_grad_pad], dim=1)
#     py_grad_padded = torch.cat([py_grad, py_grad_pad], dim=2)

#     # tot_grad's shape (B, S+1, T+1)
#     tot_grad = px_grad_padded + py_grad_padded
#     tot_grad = tot_grad.detach().cpu().permute(0, 2, 1)

#     ext = "png"  # supported types: png, ps, pdf, svg

#     x_lens = x_lens.tolist()
#     y_lens = y_lens.tolist()

#     tot_grad = tot_grad.unbind(0)
#     i = np.random.randint(0, B)  # choose random id from the batch
#     grad = tot_grad[i][: x_lens[i], : y_lens[i]]

#     filename = os.path.join(plot_dir, f"{ids[i]}.{ext}")
#     #  plt.matshow(grad.t(), origin="lower", cmap="gray")
#     plt.matshow(grad.t(), origin="lower")
#     plt.xlabel("t")
#     plt.ylabel("u")
#     plt.title(ids[i])
#     plt.savefig(filename)
#     plt.close()


# def fast_rnnt_pruned_loss(
#     enc_out,
#     dec_out,
#     targets,
#     input_lens,
#     target_lens,
#     ids,
#     curr_epoch,
#     jointer,
#     blank_index,
#     prune_range,
#     loss_scale=0.5,
#     warmup_epochs=2,
#     reduction="mean",
#     mode="simple",
#     plot_dir=None,
# ):
#     """Fast_RNNT loss, see `https://github.com/danpovey/fast_rnnt`.

#     Arguments
#     ---------
#     enc_out : torch.Tensor
#         The encoder/acoustic output, of shape [batch, audio_time_steps, vocab_size].
#     dec_out : torch.Tensor
#         The decoder/language output, of shape [batch, target_subwords_len+1, vocab_size]
#     targets : torch.Tensor
#         Target tensor, without any blanks, of shape [batch, target_subwords_len].
#     input_lens : torch.Tensor
#         Length of each utterance, of shape [batch, max_utterance_length].
#     target_lens : torch.Tensor
#         Length of each target sequence, of shape [batch, max_target_length].
#     ids: list(str)
#         A list of utterance ids in the batch. It's use mainly for plotting
#         the total fake gradient.
#     curr_epoch: int
#         The index of the current epoch, starting from 1.
#     jointer:
#         The jointer network.
#     blank_index : int
#         The location of the blank symbol among the label indices.
#     prune_range: int
#         How many symbols to keep for every frame (default = 5).
#     loss_scale: float
#         The scale for the `mode` loss. If `mode=simple`, then it's the scale
#         for the simple loss (default = 0.5).
#     warmup_epochs: int
#         The number of warmup epochs where we only learn the `mode` loss. After
#         the warmup_epochs, we are going to learn both the `mode` loss & the
#         pruned_loss.
#     reduction : str
#         Specifies the reduction to apply to the output.
#         Options: `none` | `mean` | `sum`
#     mode: str
#         The loss function type that will be applied.
#         Options: `simple` | `smoothed`
#     """
#     try:
#         import fast_rnnt
#     except ImportError:
#         err_msg = "cannot import fast_rnnt.\n"
#         err_msg += "=============================\n"
#         err_msg += "You can install Fast_RNNT using pip:\n"
#         err_msg += "pip install fast_rnnt\n"
#         raise ImportError(err_msg)

#     B, T = targets.shape  # batch_size, target_subwords_len
#     S = enc_out.size(1)  # audio_time_steps
#     boundary = torch.zeros((B, 4), dtype=torch.int64)
#     boundary[:, 2] = (target_lens * T).round().int()
#     boundary[:, 3] = (input_lens * S).round().int()
#     boundary = boundary.to(enc_out.device)

#     if mode == "simple":
#         mode_loss, (px_grad, py_grad) = fast_rnnt.rnnt_loss_simple(
#             am=enc_out,
#             lm=dec_out,
#             symbols=targets,
#             termination_symbol=blank_index,
#             boundary=boundary,
#             reduction=reduction,
#             return_grad=True,
#         )
#     elif mode == "smoothed":
#         mode_loss, (px_grad, py_grad) = fast_rnnt.rnnt_loss_smoothed(
#             am=enc_out,
#             lm=dec_out,
#             symbols=targets,
#             termination_symbol=blank_index,
#             lm_only_scale=0.5,
#             am_only_scale=0.5,
#             boundary=boundary,
#             reduction=reduction,
#             return_grad=True,
#         )
#     else:
#         raise ValueError(
#             f"Undefined option: {mode}.\n"
#             + "Available modes are:\n`simple`, `smoothed`."
#         )
#     # PLOT ALIGNMENT (10%)
#     if plot_dir and np.random.random() >= 0.9:
#         save_and_plot_tot_grad(
#             px_grad, py_grad, ids, boundary[:, 3], boundary[:, 2], plot_dir
#         )

#     # (NOTE) prune_range: min(prune_range, target_subwords_len)
#     # ranges: [batch_size, audio_time_steps, prune_range]
#     ranges = fast_rnnt.get_rnnt_prune_ranges(
#         px_grad=px_grad,
#         py_grad=py_grad,
#         boundary=boundary,
#         s_range=prune_range,
#     )

#     # enc_out_pruned: [batch_size, audio_time_steps, prune_range, vocab_size]
#     # dec_out_pruned: [batch_size, audio_time_steps, prune_range, vocab_size]
#     enc_out_pruned, dec_out_pruned = fast_rnnt.do_rnnt_pruning(
#         am=enc_out, lm=dec_out, ranges=ranges
#     )

#     # logits: [batch_size, audio_time_steps, prune_range, vocab_size]
#     logits = jointer(enc_out_pruned, dec_out_pruned)
#     pruned_loss = fast_rnnt.rnnt_loss_pruned(
#         logits=logits,
#         symbols=targets,
#         ranges=ranges,
#         termination_symbol=blank_index,
#         boundary=boundary,
#         reduction=reduction,
#     )
#     warmup = curr_epoch / warmup_epochs
#     pruned_loss_scale = (
#         0.0 if warmup < 1.0 else (0.1 if warmup > 1.0 and warmup < 2.0 else 1.0)
#     )
#     loss = loss_scale * mode_loss + pruned_loss_scale * pruned_loss
#     return loss


# #region transducer Loss

class TransducerLoss(Module):
    """
    This class implements the Transduce loss computation with forward-backward algorithm.
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf

    The TransducerLoss(nn.Module) use Transducer(autograd.Function)
    to compute the forward-backward loss and gradients.

    Input tensors must be on a cuda device.

    Arguments
    ---------
    blank : int
        Token to use as blank token.
    reduction : str
        Type of reduction to use, default "mean"

    Example
    -------
    >>> import torch
    >>> loss = TransducerLoss(blank=0)
    >>> logits = torch.randn((1,2,3,5)).cuda().requires_grad_()
    >>> labels = torch.Tensor([[1,2]]).cuda().int()
    >>> act_length = torch.Tensor([2]).cuda().int()
    >>> # U = label_length+1
    >>> label_length = torch.Tensor([2]).cuda().int()
    >>> l = loss(logits, labels, act_length, label_length)
    >>> l.backward()
    """

    def __init__(self, blank=0, reduction="mean",c_fast_emit=0):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.loss = Transducer.apply
        self.c_fast_emit = c_fast_emit
        try:
            cuda.cuda_paths
        except ImportError:
            err_msg = "cannot import numba. To use Transducer loss\n"
            err_msg += "=============================\n"
            err_msg += "If you use your localhost:\n"
            err_msg += "pip install numba\n"
            err_msg += (
                "export NUMBAPRO_LIBDEVICE='/usr/local/cuda/nvvm/libdevice/' \n"
            )
            err_msg += "export NUMBAPRO_NVVM='/usr/local/cuda/nvvm/lib64/libnvvm.so' \n"
            err_msg += "================================ \n"
            err_msg += "If you use conda:\n"
            err_msg += "conda install numba cudatoolkit=XX (XX is your cuda toolkit version)"
            raise ImportError(err_msg)

    def forward(self, logits, labels, T, U):
        """Computes the transducer loss."""
        # Transducer.apply function take log_probs tensor.
        if all(t.is_cuda for t in (logits, labels, T, U)):
            log_probs = logits.log_softmax(-1)
            return self.loss(
                log_probs, labels, T, U, self.blank, self.reduction,self.c_fast_emit
            )
        else:
            raise ValueError(
                f"Found inputs tensors to be on {[logits.device, labels.device, T.device, U.device]} while needed to be on a 'cuda' device to use the transducer loss."
            )
        

def transducer_loss(
    logits,
    targets,
    input_lens,
    target_lens,
    blank_index,
    reduction="mean",
    c_fast_emit=0
):
    """
    Arguments
    ---------
    logits : torch.Tensor
        Predicted tensor, of shape [batch, maxT, maxU, num_labels].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len].
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the label indices.
    reduction : str
        Specifies the reduction to apply to the output: 'mean' | 'batchmean' | 'sum'.

    Returns
    -------
    The computed transducer loss.
    """
    input_lens = (input_lens * logits.shape[1]).round().int()
    target_lens = (target_lens * targets.shape[1]).round().int()

    # Transducer.apply function take log_probs tensor.
    log_probs = logits.log_softmax(-1)
    return Transducer.apply(
        log_probs, targets, input_lens, target_lens, blank_index, reduction,c_fast_emit
    )