import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

from torch import nn
import copy


if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor
    

def match_seq_len(q_seqs, r_seqs, seq_len, pad_val=-1):
    proc_q_seqs = []
    proc_r_seqs = []

    for q_seq, r_seq in zip(q_seqs, r_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])

            i += seq_len + 1

        proc_q_seqs.append(
            np.concatenate(
                [
                    q_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
        proc_r_seqs.append(
            np.concatenate(
                [
                    r_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )

    return proc_q_seqs, proc_r_seqs


def match_seq_len_for_SEKT(q_seqs, s_seqs, rq_seqs, rs_seqs, seq_len, pad_val=-1):

    proc_q_seqs = []
    proc_s_seqs = []
    proc_rq_seqs = []
    proc_rs_seqs = []

    for q_seq, s_seq, rq_seq, rs_seq in zip(q_seqs, s_seqs, rq_seqs, rs_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq):
            proc_q_seqs.append(q_seq[i:i + seq_len + 1])
            proc_s_seqs.append(s_seq[i:i + seq_len + 1])
            proc_rq_seqs.append(rq_seq[i:i + seq_len + 1])
            proc_rs_seqs.append(rs_seq[i:i + seq_len + 1])

            i += seq_len + 1

        proc_q_seqs.append(
            np.concatenate(
                [
                    q_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
        proc_s_seqs.append(
            np.concatenate(
                [
                    s_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(s_seq)))
                ]
            )
        )
        proc_rq_seqs.append(
            np.concatenate(
                [
                    rq_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(rq_seq)))
                ]
            )
        )
        proc_rs_seqs.append(
            np.concatenate(
                [
                    rs_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(rs_seq)))
                ]
            )
        )

    return proc_q_seqs, proc_s_seqs, proc_rq_seqs, proc_rs_seqs


def collate_fn(batch, pad_val=-1):

    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []

    for q_seq, r_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs


def collate_fn_for_SEKT(batch, pad_val=-1):

    q_seqs = []
    s_seqs = []
    rq_seqs = []
    rs_seqs = []

    qshft_seqs = []
    sshft_seqs = []
    rqshft_seqs = []
    rsshft_seqs = []

    for q_seq, s_seq, rq_seq, rs_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1]))
        s_seqs.append(FloatTensor(s_seq[:-1]))
        rq_seqs.append(FloatTensor(rq_seq[:-1]))
        rs_seqs.append(FloatTensor(rs_seq[:-1]))

        qshft_seqs.append(FloatTensor(q_seq[1:]))
        sshft_seqs.append(FloatTensor(s_seq[1:]))
        rqshft_seqs.append(FloatTensor(rq_seq[1:]))
        rsshft_seqs.append(FloatTensor(rs_seq[1:]))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    s_seqs = pad_sequence(
        s_seqs, batch_first=True, padding_value=pad_val
    )

    rq_seqs = pad_sequence(
        rq_seqs, batch_first=True, padding_value=pad_val
    )
    rs_seqs = pad_sequence(
        rs_seqs, batch_first=True, padding_value=pad_val
    )

    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    sshft_seqs = pad_sequence(
        sshft_seqs, batch_first=True, padding_value=pad_val
    )
    rqshft_seqs = pad_sequence(
        rqshft_seqs, batch_first=True, padding_value=pad_val
    )
    rsshft_seqs = pad_sequence(
        rsshft_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, s_seqs, rq_seqs, rs_seqs, qshft_seqs, sshft_seqs, rqshft_seqs, rsshft_seqs = \
        q_seqs * mask_seqs, s_seqs * mask_seqs, \
        rq_seqs * mask_seqs, rs_seqs * mask_seqs, \
        qshft_seqs * mask_seqs, sshft_seqs * mask_seqs, \
        rqshft_seqs * mask_seqs, rsshft_seqs * mask_seqs

    return q_seqs, s_seqs, rq_seqs, rs_seqs, qshft_seqs, sshft_seqs, rqshft_seqs, rsshft_seqs, mask_seqs
