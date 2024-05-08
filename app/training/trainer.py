from transformers import Trainer
import wandb
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        head1_output, head2_output, head3_output, head4_output, head5_output, rerank_output = model(**inputs)

        # loss
        head1_loss, head2_loss, head3_loss, head4_loss, head5_loss, rerank_loss = head1_output.loss, head2_output.loss, head3_output.loss, head4_output.loss, head5_output.loss, rerank_output.loss
        final_loss = head1_loss + head2_loss + head3_loss + head4_loss + head5_loss + head1_loss + rerank_loss
        # wandb.log({"head1_loss(group)": head1_loss, "head2_loss(attribute)": head2_loss,
        #            "head3_loss(concentration)": head3_loss,
        #            "head4_loss(pack)": head4_loss, "head5_loss(packsize)": head5_loss,
        #            "rerank_loss": rerank_loss, 'combined_loss': final_loss})

        return (final_loss, (head1_output, head2_output, head3_output, head4_output, head5_output,
                             rerank_output)) if return_outputs else final_loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            # labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            labels = (inputs['labels'], inputs['head1_labels'], inputs['head2_labels'], inputs['head3_labels'],
                      inputs['head4_labels'], inputs['head5_labels'])
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
