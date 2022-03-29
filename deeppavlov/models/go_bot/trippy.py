# Copyright 2021 Neural Networks and Deep Learning lab, MIPT
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
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from pathlib import Path

import torch
from overrides import overrides
from transformers.modeling_bert import BertConfig
from transformers import BertTokenizerFast

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.models.go_bot.nlg.nlg_manager import NLGManagerInterface
from deeppavlov.models.go_bot.policy.dto.policy_prediction import PolicyPrediction
from deeppavlov.models.go_bot.trippy_bert_for_dst import BertForDST
from deeppavlov.models.go_bot.trippy_preprocessing import prepare_trippy_data, get_turn, batch_to_device
from deeppavlov.models.go_bot.trippy_dst import TripPyDST

logger = getLogger(__name__)

@register('trippy')
class TripPy(TorchModel):
    """
    Go-bot architecture based on https://arxiv.org/abs/2005.02877.

    Parameters:
        nlg_manager: DeepPavlov NLGManager responsible for answer generation
        save_path: Where to save the model
        slot_names: Names of all slots present in the data
        class_types: TripPy Class types - Predefined to most commonly used; Add True&False if slots which can take on those values
        pretrained_bert: bert-base-uncased or full path to pretrained model
        bert_config: Can be path to a file in case different from bert-base-uncased config
        optimizer_parameters: dictionary with optimizer's parameters, e.g. {'lr': 0.1, 'weight_decay': 0.001, 'momentum': 0.9}
        clip_norm: Clip gradients by norm
        max_seq_length: Max sequence length of an entire dialog. Defaults to TripPy 180 default. Too long examples will be logged.
        class_loss_ratio: The ratio applied on class loss in total loss calculation.
            Should be a value in [0.0, 1.0].
            The ratio applied on token loss is (1-class_loss_ratio)/2.
            The ratio applied on refer loss is (1-class_loss_ratio)/2.
        token_loss_for_nonpointable: Whether the token loss for classes other than copy_value contribute towards total loss.
        refer_loss_for_nonpointable: Whether the refer loss for classes other than refer contribute towards total loss.
        class_aux_feats_inform: Whether or not to use the identity of informed slots as auxiliary features for class prediction.
        class_aux_feats_ds: Whether or not to use the identity of slots in the current dialog state as auxiliary featurs for class prediction.
        database: Optional database which will be queried by make_api_call by default
        make_api_call: Optional function to replace default api calling
        fill_current_state_with_db_results: Optional function t replace default db result filling
        debug: Turn on debug mode to get logging information on input examples & co
    """
    def __init__(self,
                 nlg_manager: NLGManagerInterface,
                 save_path: str,
                 slot_names: List = [],
                 class_types: List = ["none", "dontcare", "copy_value", "inform"],
                 pretrained_bert: str = "bert-base-uncased",
                 bert_config: str = "bert-base-uncased",
                 optimizer_parameters: dict = {"lr": 1e-5, "eps": 1e-6},
                 clip_norm: float = 1.0,
                 max_seq_length: int = 180,
                 dropout_rate: float = 0.3,
                 heads_dropout: float = 0.0,
                 class_loss_ratio: float = 0.8,
                 token_loss_for_nonpointable: bool = False,
                 refer_loss_for_nonpointable: bool = False,
                 class_aux_feats_inform: bool = True,
                 class_aux_feats_ds: bool = True,
                 database: Component = None,
                 make_api_call: Callable = None,
                 fill_current_state_with_db_results: Callable = None,
                 debug: bool = False,
                 **kwargs) -> None:

        self.nlg_manager = nlg_manager
        self.save_path = save_path
        self.max_seq_length = max_seq_length
        self.slot_names = slot_names
        self.class_types = class_types
        self.debug = debug

        # BertForDST Configuration
        self.pretrained_bert = pretrained_bert
        self.config = BertConfig.from_pretrained(bert_config)
        self.config.dst_dropout_rate = dropout_rate
        self.config.dst_heads_dropout_rate = heads_dropout
        self.config.dst_class_loss_ratio = class_loss_ratio
        self.config.dst_token_loss_for_nonpointable = token_loss_for_nonpointable
        self.config.dst_refer_loss_for_nonpointable = refer_loss_for_nonpointable
        self.config.dst_class_aux_feats_inform = class_aux_feats_inform
        self.config.dst_class_aux_feats_ds = class_aux_feats_ds
        self.config.dst_slot_list = self.slot_names
        self.config.dst_class_types = class_types
        self.config.dst_class_labels = len(class_types)

        self.config.num_actions = nlg_manager.num_of_known_actions()

        self.clip_norm = clip_norm

        # Get TripPy-specific Dialogue State Tracker
        self.dst = TripPyDST(slot_names, class_types, database, make_api_call, fill_current_state_with_db_results)

        super().__init__(save_path=save_path,  
                        optimizer_parameters=optimizer_parameters,
                        **kwargs)

    @overrides
    def load(self, fname=None):
        """
        Loads BERTForDST. Note that it only supports bert-X huggingface weights. (RoBERTa & co are not supported.)
        """


        if fname is not None:
            self.load_path = fname

        if self.pretrained_bert:
            self.model = BertForDST.from_pretrained(
                self.pretrained_bert, config=self.config)
            self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_bert)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        # Data Parallelism in case of Multi-GPU setup
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        if self.load_path:
            logger.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                logger.info(f"Load path {weights_path} exists.")
                logger.info(f"Initializing `{self.__class__.__name__}` from saved.")

                # now load the weights, optimizer from saved
                logger.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                logger.info(f"Init from scratch. Load path {weights_path} does not exist.")


    def __call__(self,
                 batch: Union[List[List[dict]], List[str]],
                 user_ids: Optional[List] = None) -> List:
        """
        Model invocation.

        Args:
            batch: batch of dialogue data or list of strings
            user_ids: Id that identifies the user # Check bocks

        Returns:
            results: list of model answers
        """
        # Turns off dropout
        self.model.eval()

        if not(isinstance(batch[0], list)):
            # User inference - Just one dialogue
            diag_batch = [
                [{"text": text, "intents": [{"act": None, "slots": None}]} for text in batch]
                ]
        else:
            diag_batch = batch
            # At validation reset for every call
            self.dst.reset()

        dialogue_results = []
        for diag_id, dialogue in enumerate(diag_batch):

            turn_results = []
            for turn_id, turn in enumerate(dialogue):
                # Reset dialogue state if no dialogue state yet or the dialogue is empty (i.e. its a new dialogue)
                #if (self.ds_logits is None) or (diag_id >= len(self.batch_dialogues_utterances_contexts_info)):
                if (self.dst.ds_logits is None) or (diag_id >= len(self.dst.batch_dialogues_utterances_contexts_info)):
                    self.dst.reset()
                    diag_id = 0

                # Append context to the dialogue
                self.dst.batch_dialogues_utterances_contexts_info[diag_id].append(turn)

                # Update Database
                self.dst.update_ground_truth_db_result_from_context(turn)

                # Preprocess inputs
                trippy_input, features = prepare_trippy_data(self.dst.batch_dialogues_utterances_contexts_info,
                                                            self.dst.batch_dialogues_utterances_responses_info,
                                                            self.tokenizer,
                                                            self.slot_names,
                                                            self.class_types,
                                                            self.nlg_manager,
                                                            max_seq_length=self.max_seq_length,
                                                            debug=self.debug)

                # Take only the last turn - as we already know the previous ones; We need to feed them one by one to update the ds
                last_turn = get_turn(trippy_input, index=-1)

                # Only take them from the last turn
                input_ids_unmasked = [features[-1].input_ids_unmasked]
                inform = [features[-1].inform]

                # Update data-held dialogue state based on new logits
                last_turn["diag_state"] = self.dst.ds_logits

                # Move to correct device
                last_turn = batch_to_device(last_turn, self.device)

                # If there are no slots, remove not needed data
                if not(self.slot_names):
                    last_turn["start_pos"] = None
                    last_turn["end_pos"] = None
                    last_turn["inform_slot_id"] = None
                    last_turn["refer_id"] = None
                    last_turn["class_label_id"] = None
                    last_turn["diag_state"] = None

                # Run the turn through the model
                with torch.no_grad():
                    outputs = self.model(**last_turn)

                # Update dialogue state logits
                for slot in self.slot_names:
                    updates = outputs[2][slot].max(1)[1].cpu()
                    for i, u in enumerate(updates):
                        if u != 0:
                            self.dst.ds_logits[slot][i] = u

                # Update self.ds (dialogue state) slotfilled values based on logits
                self.dst.update_ds(outputs[2],
                               outputs[3],
                               outputs[4],
                               outputs[5],
                               input_ids_unmasked,
                               inform,
                               self.tokenizer)

                # Wrap predicted action (outputs[6]) into a PolicyPrediction
                policy_prediction = PolicyPrediction(
                    outputs[6].cpu().numpy(), None, None, None)

                # Fill DS with Database results if there are any
                self.dst.fill_current_state_with_db_results()

                # NLG based on predicted action & dialogue state
                response = self.nlg_manager.decode_response(None,
                                                            policy_prediction,
                                                            self.dst.ds)

                # Add system response to responses for possible next round
                self.dst.batch_dialogues_utterances_responses_info[diag_id].insert(
                    -1, {"text": response, "act": None})

                turn_results.append(response)

            dialogue_results.append(turn_results)
        
        # At real-time interaction make an actual api call if this is the action predicted
        if (not(isinstance(batch[0], list))) and (policy_prediction.predicted_action_ix == self.nlg_manager.get_api_call_action_id()):
            self.dst.make_api_call()
            # Call TripPy again with the same user text - This is how it is done in the DSTC2 Training Data
            # Note that now the db_results are updated and the last system response has been api_call
            # Then return the last two system responses of the form [[api_call..., I have found...]]
            dialogue_results[-1].append(self(batch)[-1][-1])
            return dialogue_results
            
        # Return NLG generated responses
        return dialogue_results

    def train_on_batch(self,
                       batch_dialogues_utterances_features: List[List[dict]],
                       batch_dialogues_utterances_targets: List[List[dict]]) -> dict:
        """
        Train model on given batch.

        Args:
            batch_dialogues_utterances_features:
            batch_dialogues_utterances_targets: 

        Returns:
            dict with loss value
        """
        # Turns on dropout
        self.model.train()
        # Zeroes grads
        self.model.zero_grad()
        batch, features = prepare_trippy_data(batch_dialogues_utterances_features,
                                              batch_dialogues_utterances_targets,
                                              self.tokenizer,
                                              self.slot_names,
                                              self.class_types,
                                              self.nlg_manager,
                                              self.max_seq_length,
                                              debug=self.debug)
        # Move to correct device
        batch = batch_to_device(batch, self.device)

        if not(self.slot_names):
            batch["start_pos"] = None
            batch["end_pos"] = None
            batch["inform_slot_id"] = None
            batch["refer_id"] = None
            batch["class_label_id"] = None
            batch["diag_state"] = None

        # Feed through model
        outputs = self.model(**batch)

        # Backpropagation
        loss = outputs[0]
        action_loss = outputs[7]

        # Average device results in case of multi-gpu setup
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
            action_loss = action_loss.mean()

        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
        return {"total_loss": loss.cpu().item(), "action_loss": action_loss.cpu().item()}
