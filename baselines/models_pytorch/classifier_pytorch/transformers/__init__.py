__version__ = "2.1.1"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Files and general utilities
from .file_utils import (TRANSFORMERS_CACHE, PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME,
                         is_tf_available, is_torch_available)

# Tokenizers
from .tokenization_utils import (PreTrainedTokenizer)
from .tokenization_auto import AutoTokenizer
from .tokenization_bert import BertTokenizer, BasicTokenizer, WordpieceTokenizer
from .tokenization_openai import OpenAIGPTTokenizer
from .tokenization_transfo_xl import (TransfoXLTokenizer, TransfoXLCorpus)
from .tokenization_gpt2 import GPT2Tokenizer
from .tokenization_ctrl import CTRLTokenizer
from .tokenization_xlnet import XLNetTokenizer, SPIECE_UNDERLINE
from .tokenization_xlm import XLMTokenizer
from .tokenization_roberta import RobertaTokenizer
from .tokenization_distilbert import DistilBertTokenizer

# Configurations
from .configuration_utils import PretrainedConfig
from .configuration_auto import AutoConfig
from .configuration_bert import BertConfig, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_openai import OpenAIGPTConfig, OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_transfo_xl import TransfoXLConfig, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_gpt2 import GPT2Config, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_ctrl import CTRLConfig, CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlnet import XLNetConfig, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_ctrl import CTRLConfig, CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_xlm import XLMConfig, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_roberta import RobertaConfig, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP
from .configuration_distilbert import DistilBertConfig, DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

# Modeling
if is_torch_available():
    from .modeling_utils import (PreTrainedModel, prune_layer, Conv1D)
    from .modeling_auto import (AutoModel, AutoModelForSequenceClassification, AutoModelForQuestionAnswering,
                                AutoModelWithLMHead)

    from .modeling_bert import (BertPreTrainedModel, BertModel, BertForPreTraining,
                                BertForMaskedLM, BertForNextSentencePrediction,
                                BertForSequenceClassification, BertForMultipleChoice,
                                BertForTokenClassification, BertForQuestionAnswering,
                                load_tf_weights_in_bert, BERT_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_openai import (OpenAIGPTPreTrainedModel, OpenAIGPTModel,
                                OpenAIGPTLMHeadModel, OpenAIGPTDoubleHeadsModel,
                                load_tf_weights_in_openai_gpt, OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_transfo_xl import (TransfoXLPreTrainedModel, TransfoXLModel, TransfoXLLMHeadModel,
                                    load_tf_weights_in_transfo_xl, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_gpt2 import (GPT2PreTrainedModel, GPT2Model,
                                GPT2LMHeadModel, GPT2DoubleHeadsModel,
                                load_tf_weights_in_gpt2, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_ctrl import (CTRLPreTrainedModel, CTRLModel,
                                CTRLLMHeadModel,
                                CTRL_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_xlnet import (XLNetPreTrainedModel, XLNetModel, XLNetLMHeadModel,
                                XLNetForSequenceClassification, XLNetForMultipleChoice,
                                XLNetForQuestionAnsweringSimple, XLNetForQuestionAnswering,
                                load_tf_weights_in_xlnet, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_xlm import (XLMPreTrainedModel , XLMModel,
                            XLMWithLMHeadModel, XLMForSequenceClassification,
                            XLMForQuestionAnswering, XLMForQuestionAnsweringSimple,
                            XLM_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_roberta import (RobertaForMaskedLM, RobertaModel,
                                RobertaForSequenceClassification, RobertaForMultipleChoice,
                                ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_distilbert import (DistilBertForMaskedLM, DistilBertModel,
                                DistilBertForSequenceClassification, DistilBertForQuestionAnswering,
                                DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP)
    from .modeling_albert import AlbertForSequenceClassification

    # Optimization
    from .optimization import (AdamW, ConstantLRSchedule, WarmupConstantSchedule, WarmupCosineSchedule,
                               WarmupCosineWithHardRestartsSchedule, WarmupLinearSchedule)
if not is_tf_available() and not is_torch_available():
    logger.warning("Neither PyTorch nor TensorFlow >= 2.0 have been found."
                   "Models won't be available and only tokenizers, configuration"
                   "and file/data utilities can be used.")
