# Software Name: attentionless-streaming-asr
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html 

import torch
import speechbrain as sb

from speechbrain.decoders.transducer import TransducerBeamSearcher
import sentencepiece


class BeamSearcher(TransducerBeamSearcher):

    def __init__(
        self,
        decode_network_lst,
        tjoint,
        classifier_network,
        blank_id,
        beam_size=4,
        nbest=5,
        lm_module=None,
        lm_weight=0.0,
        state_beam=2.3,
        expand_beam=2.3,
        have_hiddens=True,
    ):
        super().__init__(decode_network_lst,
        tjoint,
        classifier_network,
        blank_id,
        beam_size=beam_size,
        nbest=nbest,
        lm_module=lm_module,
        lm_weight=lm_weight,
        state_beam=state_beam,
        expand_beam=expand_beam)

        self.batch_frame_tokens = [] #record the decoding timestamp of each token in each batch
        self.batch_last_tokens_idx = [] #record the frame where the last token is decoded
        self.have_hiddens = have_hiddens

    def transducer_greedy_decode(
            self, tn_output, hidden_state=None, return_hidden=False
        ):
            """Transducer greedy decoder is a greedy decoder over batch which apply Transducer rules:
                1- for each time step in the Transcription Network (TN) output:
                    -> Update the ith utterance only if
                        the previous target != the new one (we save the hiddens and the target)
                    -> otherwise:
                    ---> keep the previous target prediction from the decoder

            Arguments
            ---------
            tn_output : torch.Tensor
                Output from transcription network with shape
                [batch, time_len, hiddens].
            hidden_state : (torch.Tensor, torch.Tensor)
                Hidden state to initially feed the decode network with. This is
                useful in conjunction with `return_hidden` to be able to perform
                beam search in a streaming context, so that you can reuse the last
                hidden state as an initial state across calls.
            return_hidden : bool
                Whether the return tuple should contain an extra 5th element with
                the hidden state at of the last step. See `hidden_state`.

            Returns
            -------
            Tuple of 4 or 5 elements (if `return_hidden`).

            First element: List[List[int]]
                List of decoded tokens

            Second element: torch.Tensor
                Outputs a logits tensor [B,T,1,Output_Dim]; padding
                has not been removed.

            Third element: None
                nbest; irrelevant for greedy decode

            Fourth element: None
                nbest scores; irrelevant for greedy decode

            Fifth element: Present if `return_hidden`, (torch.Tensor, torch.Tensor)
                Tuple representing the hidden state required to call
                `transducer_greedy_decode` where you left off in a streaming
                context.
            """
            #reinitialise token_positions
            self.batch_frame_tokens = [[] for _ in range(tn_output.size(0))]
            self.batch_last_tokens_idx = [0 for _ in range(tn_output.size(0))]

            hyp = {
                "prediction": [[] for _ in range(tn_output.size(0))],
                "logp_scores": [0.0 for _ in range(tn_output.size(0))],
            }
            # prepare BOS = Blank for the Prediction Network (PN)
            input_PN = (
                torch.ones(
                    (tn_output.size(0), 1),
                    device=tn_output.device,
                    dtype=torch.int32,
                )
                * self.blank_id
            )

            if hidden_state is None:
                # First forward-pass on PN
                out_PN, hidden = self._forward_PN(input_PN, self.decode_network_lst)
            else:
                out_PN, hidden = hidden_state

            # For each time step
            for t_step in range(tn_output.size(1)):
                # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
                count = 0
                while count <= 5: #avoid infinite loop
                    log_probs = self._joint_forward_step(
                        tn_output[:, t_step, :].unsqueeze(1).unsqueeze(1),
                        out_PN.unsqueeze(1),
                    )
                    # Sort outputs at time
                    logp_targets, positions = torch.max(
                        log_probs.squeeze(1).squeeze(1), dim=1
                    )
                    if self.have_hiddens:
                        # Batch hidden update
                        # Initialize have_update_hyp to avoid undefined variable error
                        have_update_hyp = []
                        for i in range(positions.size(0)):
                            # Update hiddens only if
                            # 1- current prediction is non blank
                            if positions[i].item() != self.blank_id:
                                hyp["prediction"][i].append(positions[i].item())
                                hyp["logp_scores"][i] += logp_targets[i]
                                input_PN[i][0] = positions[i]
                                have_update_hyp.append(i)

                                self.batch_frame_tokens[i].append([positions[i].item()])
                                self.batch_last_tokens_idx[i] = (t_step+1)*40 #frame rate is 40ms
                            else:
                                self.batch_frame_tokens[i].append([])

                        if len(have_update_hyp) > 0:
                            # Select sentence to update
                            # And do a forward steps + generated hidden
                            (
                                selected_input_PN,
                                selected_hidden,
                            ) = self._get_sentence_to_update(
                                have_update_hyp, input_PN, hidden
                            )
                            selected_out_PN, selected_hidden = self._forward_PN(
                                selected_input_PN, self.decode_network_lst, selected_hidden
                            )
                            # update hiddens and out_PN
                            out_PN[have_update_hyp] = selected_out_PN
                            hidden = self._update_hiddens(
                                have_update_hyp, selected_hidden, hidden
                            )
                        else:
                            break
                        count += 1
                    

            ret = (
                hyp["prediction"],
                torch.Tensor(hyp["logp_scores"]).exp().mean(),
                None,
                None,
            )

            if return_hidden:
                # append the `(out_PN, hidden)` tuple to ret
                ret += (
                    (
                        out_PN,
                        hidden,
                    ),
                )

            return ret