""" PyTorch Relation Extraction LayoutLMv2 outputs """
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch


@dataclass
class RegionExtractionOutput(ModelOutput):
    """
    Region extraction model output class.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            CE loss on labels from the relations dict
        logits (`torch.FloatTensor` of shape `(batch_size, relations_length, relations_length)`):
            Prediction scores of relationsips between entities
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        entities (list of dicts of shape `(batch_size,)` where each dict contains:
            {
                'start': `torch.IntTensor` of shape `(num_entites)`,
                    Each value in the list represents the id of the token (element of range(0, len(tokens)) where the
                    entity starts
                'end': `torch.IntTensor` of shape `(num_entites)`,
                    Each value in the list represents the id of the token (element of range(0, len(tokens)) where the
                    entity ends
                'label': `torch.IntTensor` of shape `(num_entites)`
                    Each value in the list represents the label (as an int) of the entity
            }
        relations (list of dicts of shape `(batch_size,)` where each dict contains:
            {
                'head': `torch.IntTensor` of shape `(up to num_entites^2)`,
                    Each value in the list represents the key of a different relation. A value can be used to map to
                    the entity list as it tells you what index to inspect in any of the lists inside the entities dict
                    (reps the id of the entity `(element of range(0, len(entities)`)
                'tail': `torch.IntTensor` of shape `(up to num_entites^2)`,
                    Each value in the list represents the value of a different relation. A value can be used to map to
                    the entity list as it tells you what index to inspect in any of the lists inside the entities dict
                    (reps the id of the entity `(element of range(0, len(entities)`)
                'start_index': `torch.IntTensor` of shape `(up to num_entites^2)`,
                    Each value in this list represents the start index (element of range(0, len(tokens)) for the
                    combined head and tail entities e.g. `min(entities['start']['head'], entities['start']['tail'])`
                'end_index': `torch.IntTensor` of shape `(up to num_entites^2)`,
                    Each value in this list represents the end index (element of range(0, len(tokens)) for the
                    combined head and tail entities e.g. `min(entities['end']['head'], entities['end']['tail'])`
            }
        pred_relations (list of lists of shape `(batch_size, pred_relations)` where each element is a dict containing:
            {
                'head': `tuple` of `(start_token_index, end_token_index)`,
                    This value shows gets the start and end tokens of the entity for which the relation predicted to
                    be the key
                'head_id': `int`,
                    This value can be used to map to the entity list as it tells you what index to inspect in any of
                    the lists inside the entities dict(reps the id of the entity `(element of range(0, len(entities)`)
                'head_type': `int`,
                    This value is set to the label value of the corrosponding entity
                'tail': `tuple` of `(start_token_index, end_token_index)`,
                    This value shows gets the start and end tokens of the entity for which the relation predicted to
                    be the value
                'tail_id': `int`,
                    This value can be used to map to the entity list as it tells you what index to inspect in any of
                    the lists inside the entities dict(reps the id of the entity `(element of range(0, len(entities)`)
                'tail_type': `int`,
                    This value is set to the label value of the corrosponding entity
                'type': `int`,
                    This value is set to `1` for a predicted relation
            }    
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    entities: Optional[Dict] = None
    relations: Optional[Dict] = None
    pred_relations: Optional[Dict] = None