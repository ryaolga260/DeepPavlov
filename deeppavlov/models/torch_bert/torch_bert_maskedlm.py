# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from logging import getLogger
from pathlib import Path
from typing import List, Dict, Union, Optional

import numpy as np
import torch
from overrides import overrides
from transformers import AutoModelForMaskedLM, AutoConfig
from transformers.data.processors.utils import InputFeatures

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.preprocessors.torch_transformers_preprocessor import TorchTransformersPreprocessor

logger = getLogger(__name__)


@register('torch_bert_maskedlm')
class TorchBertMaskedLM(TorchModel):
    """Masked Language model based on BERT on PyTorch.
    BERT model was trained on Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) tasks.

    Args:
        pretrained_bert: pretrained Bert checkpoint path or key title (e.g. "bert-base-uncased")
        bert_config_file: path to Bert configuration file (not used if pretrained_bert is key title)
        vocab_file: path to Bert vocabulary
        max_seq_length: max sequence length in subtokens, including ``[SEP]`` and ``[CLS]`` tokens.
            `max_seq_length` is used in Bert to compute NSP scores. Defaults to ``128``.
        do_lower_case: set ``False`` if cased is needed. Note that this will require using bert-base-cased. Defaults to ``True``.
    """

    def __init__(self, pretrained_bert: str,
                 vocab_file: str,
                 bert_config_file: Optional[str] = None,
                 max_seq_length: int = 128,
                 do_lower_case: bool = True,
                 optimizer: str = "AdamW",
                 optimizer_parameters: dict = {"lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.999), "eps": 1e-6},
                 clip_norm: Optional[float] = None,
                 save_path: Optional[str] = None,
                 **kwargs) -> None:

        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.bert_preprocessor = TorchTransformersPreprocessor(vocab_file=vocab_file, do_lower_case=do_lower_case,
                                                               max_seq_length=max_seq_length)
        self.clip_norm = clip_norm

        super().__init__(optimizer=optimizer,
                        optimizer_parameters=optimizer_parameters,
                        save_path=save_path, **kwargs)

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_bert and not Path(self.pretrained_bert).is_file():
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.pretrained_bert, output_attentions=False, output_hidden_states=False)
        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.bert_config = AutoConfig.from_json_file(str(expand_path(self.bert_config_file)))
            self.model = AutoModelForMaskedLM(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        self.model.to(self.device)


    def __call__(self, features: List[str]) -> List[str]:
        """Make prediction for given features (texts).

        Args:
            features: List of texts with [MASK] token

        Returns:
            Predicted text without [MASK] token
        """

        # Get features - i.e. input_ids, attn_mask, ttype_ids
        features = self.bert_preprocessor(features)
        
        # Convert into dict
        _input = {}
        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            _input[elem] = [getattr(f, elem) for f in features]

        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            _input[elem] = torch.cat(_input[elem], dim=0).to(self.device)


        with torch.no_grad():
            features = {key:value for (key,value) in _input.items() if key in self.model.forward.__code__.co_varnames}

            logits = self.model(**features).logits

            logits = np.argmax(logits, axis=-1)

        preds = []

        for i in range(logits.shape[0]):
            # Reduce to same length as the text input for that sentence
            index = torch.where(features["attention_mask"][i] == 0)[0][0]
            pred = self.bert_preprocessor.tokenizer.decode(logits[i,:index])
            preds.append(pred)

        return preds

    def train_on_batch(self, features: List[InputFeatures], y: Union[List[int], List[List[int]]]) -> Dict:
        """Train model on given batch.
        This method calls train_op using features and y (labels).

        Args:
            features: batch of InputFeatures with masked tokens, i.e. [MASK]
            y: batch of input_ids without [MASK] tokens, i.e. len(features.input_ids) == len(y)

        Returns:
            dict with loss and learning_rate values
        """

        _input = {}
        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            _input[elem] = [getattr(f, elem) for f in features]

        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            _input[elem] = torch.cat(_input[elem], dim=0).to(self.device)

        _input['labels'] = torch.cat(y, dim=0).to(self.device) # labels should be the exact same as input_ids, but w/o real ids instead of [MASK] ids

        self.optimizer.zero_grad()

        tokenized = {key:value for (key,value) in _input.items() if key in self.model.forward.__code__.co_varnames}

        # Token_type_id is omitted for Text Classification

        loss, logits = self.model(**tokenized)
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}
