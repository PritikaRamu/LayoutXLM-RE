from dataclasses import dataclass, fields
from collections import OrderedDict
from typing import Dict, Tuple, Optional, Any
import torch

class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.
    <Tip warning={true}>
    You can't unpack a `ModelOutput` directly. Use the [`~file_utils.ModelOutput.to_tuple`] method to convert it to a
    tuple before.
    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

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